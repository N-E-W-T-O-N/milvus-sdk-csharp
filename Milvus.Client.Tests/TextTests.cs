using Xunit;

namespace Milvus.Client.Tests;

// Milvus 2.6.4 crashes outright (process exits with SIGABRT) when two or more collections
// containing a Text field are created/inserted/loaded concurrently -- reliably reproduced by running
// this class's own tests in parallel with each other (confirmed via a minimal repro: 10 *sequential*
// create/insert/load/drop cycles with a Text field never crash it, but running this class's 6 tests
// together, which xunit v3 parallelizes by default, crashes it consistently). This is a server-side
// bug, not something the client can work around -- keep this class out of the parallel pool so its
// own tests, and the rest of the suite, aren't taken down by it.
[CollectionDefinition(nameof(TextTests), DisableParallelization = true)]
public sealed class TextTestsCollection;

[Collection(nameof(TextTests))]
public class TextTests : IAsyncLifetime
{
    private readonly MilvusClient Client;

    public TextTests(MilvusFixture milvusFixture) => Client = milvusFixture.CreateClient();

    [Fact]
    public async Task Insert_and_query_round_trip()
    {
        if (await Skip()) return;

        MilvusCollection collection = await CreateCollectionAsync(nameof(Insert_and_query_round_trip));

        await collection.CreateIndexAsync("vec", IndexType.Flat, SimilarityMetricType.L2, cancellationToken: TestContext.Current.CancellationToken);
        await collection.InsertAsync(new FieldData[]
        {
            FieldData.Create("id", new long[] { 1 }),
            FieldData.CreateText("content", new[] { "hello text world" }),
            FieldData.CreateFloatVector("vec", new ReadOnlyMemory<float>[] { new float[] { 1, 1, 1, 1 } }),
        }, cancellationToken: TestContext.Current.CancellationToken);
        await collection.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);
        await collection.WaitForCollectionLoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        var results = await collection.QueryAsync(
            "id == 1", new QueryParameters { OutputFields = { "content" } }, cancellationToken: TestContext.Current.CancellationToken);
        var field = (FieldData<string?>)Assert.Single(results, f => f.FieldName == "content");
        Assert.Equal("hello text world", Assert.Single(field.Data));

        await collection.DropAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Describe_round_trips_properties()
    {
        if (await Skip()) return;

        MilvusCollection collection = await CreateCollectionAsync(nameof(Describe_round_trips_properties));

        MilvusCollectionDescription description = await collection.DescribeAsync(TestContext.Current.CancellationToken);
        FieldSchema field = description.Schema.Fields.Single(f => f.Name == "content");

        Assert.Equal(MilvusDataType.Text, field.DataType);
        Assert.Equal(1000, field.MaxLength);
        Assert.True(field.EnableAnalyzer);

        await collection.DropAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Insert_without_max_length_fails()
    {
        if (await Skip()) return;

        // Milvus does not enforce max_length for a Text field at collection-creation time, but every
        // insert into the field then fails -- confirmed against 2.6.4. FieldSchema.CreateText requires
        // maxLength to avoid this trap; this test bypasses that via the general Create(...) overload to
        // confirm the underlying server behavior it protects against still holds.
        MilvusCollection collection = Client.GetCollection(nameof(Insert_without_max_length_fails));
        await collection.DropAsync(TestContext.Current.CancellationToken);

        await Client.CreateCollectionAsync(
            nameof(Insert_without_max_length_fails),
            new[]
            {
                FieldSchema.Create<long>("id", isPrimaryKey: true),
                FieldSchema.Create("content", MilvusDataType.Text),
                FieldSchema.CreateFloatVector("vec", 4),
            }, cancellationToken: TestContext.Current.CancellationToken);

        MilvusException exception = await Assert.ThrowsAsync<MilvusException>(() =>
            collection.InsertAsync(new FieldData[]
            {
                FieldData.Create("id", new long[] { 1 }),
                FieldData.CreateText("content", new[] { "hello" }),
                FieldData.CreateFloatVector("vec", new ReadOnlyMemory<float>[] { new float[] { 1, 1, 1, 1 } }),
            }, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("max length", exception.Message, StringComparison.OrdinalIgnoreCase);

        await collection.DropAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Rejects_as_primary_key()
    {
        if (await Skip()) return;

        MilvusCollection collection = Client.GetCollection(nameof(Rejects_as_primary_key));
        await collection.DropAsync(TestContext.Current.CancellationToken);

        await Assert.ThrowsAsync<MilvusException>(() =>
            Client.CreateCollectionAsync(
                nameof(Rejects_as_primary_key),
                new[]
                {
                    FieldSchema.Create("id", MilvusDataType.Text, isPrimaryKey: true),
                    FieldSchema.CreateFloatVector("vec", 4),
                }, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Rejects_default_value()
    {
        if (await Skip()) return;

        MilvusCollection collection = Client.GetCollection(nameof(Rejects_default_value));
        await collection.DropAsync(TestContext.Current.CancellationToken);

        // FieldSchema.CreateText has no defaultValue parameter; go through the general overload to confirm
        // the server (not just the client) rejects it -- consistent with Milvus's documented "no default
        // values" restriction for Text fields. Milvus itself doesn't get a chance to weigh in here: our
        // own FieldSchema.ToGrpc-equivalent conversion rejects it client-side first.
        await Assert.ThrowsAsync<ArgumentException>(() =>
            Client.CreateCollectionAsync(
                nameof(Rejects_default_value),
                new[]
                {
                    FieldSchema.Create<long>("id", isPrimaryKey: true),
                    FieldSchema.Create("content", MilvusDataType.Text, nullable: true, defaultValue: "fallback"),
                    FieldSchema.CreateFloatVector("vec", 4),
                }, cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task TEXT_MATCH_is_currently_rejected()
    {
        if (await Skip()) return;

        // Unlike VarChar (see TextMatchTests), Milvus rejects any filter expression -- TEXT_MATCH
        // included -- against a Text field outright, even with EnableMatch and EnableAnalyzer both set.
        // Confirmed against 2.6.4: "filter on text field (...) is not supported yet". This test fails
        // loudly if that ever changes, so the docs here don't go stale silently.
        MilvusCollection collection = Client.GetCollection(nameof(TEXT_MATCH_is_currently_rejected));
        await collection.DropAsync(TestContext.Current.CancellationToken);

        await Client.CreateCollectionAsync(
            nameof(TEXT_MATCH_is_currently_rejected),
            new[]
            {
                FieldSchema.Create<long>("id", isPrimaryKey: true),
                FieldSchema.CreateText("content", 1000, enableAnalyzer: true, enableMatch: true),
                FieldSchema.CreateFloatVector("vec", 4),
            }, cancellationToken: TestContext.Current.CancellationToken);

        await collection.CreateIndexAsync("vec", IndexType.Flat, SimilarityMetricType.L2, cancellationToken: TestContext.Current.CancellationToken);
        await collection.LoadAsync(cancellationToken: TestContext.Current.CancellationToken);
        await collection.WaitForCollectionLoadAsync(cancellationToken: TestContext.Current.CancellationToken);

        MilvusException exception = await Assert.ThrowsAsync<MilvusException>(() =>
            collection.QueryAsync("TEXT_MATCH(content, 'fox')", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("not supported", exception.Message, StringComparison.OrdinalIgnoreCase);

        await collection.DropAsync(TestContext.Current.CancellationToken);
    }

    private async Task<MilvusCollection> CreateCollectionAsync(string name)
    {
        MilvusCollection collection = Client.GetCollection(name);
        await collection.DropAsync(TestContext.Current.CancellationToken);

        await Client.CreateCollectionAsync(
            name,
            new[]
            {
                FieldSchema.Create<long>("id", isPrimaryKey: true),
                FieldSchema.CreateText("content", 1000, enableAnalyzer: true),
                FieldSchema.CreateFloatVector("vec", 4),
            }, cancellationToken: TestContext.Current.CancellationToken);

        return collection;
    }

    private async Task<bool> Skip() => await Client.GetParsedMilvusVersion() < new Version(2, 6);

    public ValueTask InitializeAsync() => ValueTask.CompletedTask;

    public ValueTask DisposeAsync()
    {
        Client.Dispose();
        return ValueTask.CompletedTask;
    }
}
