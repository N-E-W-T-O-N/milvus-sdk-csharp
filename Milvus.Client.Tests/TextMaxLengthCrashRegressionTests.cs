using Xunit;

namespace Milvus.Client.Tests;

// Regression test for the server crash documented on TextTests's TextTestsCollection: a collection
// with a Text field lacking max_length makes Milvus 2.6.4's streaming-node flusher panic and take the
// whole server process down, on a delayed WAL-recovery pass rather than synchronously. Tracked upstream
// at milvus-io/milvus#53291.
//
// This test deliberately creates that exact shape, so it runs against its own IsolatedMilvusFixture
// container rather than MilvusFixture's assembly-wide shared one -- if the server does crash here (or
// once the delayed panic eventually fires on this container), it only takes down this one test class,
// not the other 280+ tests sharing the normal container.
public class TextMaxLengthCrashRegressionTests : IClassFixture<IsolatedMilvusFixture>, IDisposable
{
    private readonly MilvusClient? Client;

    public TextMaxLengthCrashRegressionTests(IsolatedMilvusFixture fixture)
        => Client = fixture.IsAvailable ? fixture.CreateClient() : null;

    public void Dispose() => Client?.Dispose();

    [Fact]
    public async Task Insert_without_max_length_fails()
    {
        // IsolatedMilvusFixture itself decides, from the MILVUS_IMAGE tag alone, whether this version
        // is new enough to bother starting a container for -- Text (and this crash) is 2.6+. A null
        // Client here means it decided no, so there's nothing to test on this CI image.
        if (Client is null)
        {
            return;
        }

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
}
