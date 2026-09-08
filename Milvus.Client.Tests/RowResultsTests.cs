using Xunit;

namespace Milvus.Client.Tests;

// See TextTests's TextTestsCollection comment for the actual root cause of the server crashes that
// motivated this: a Text field created without max_length, not concurrency. This class never creates
// that shape, so it isn't known to trigger anything itself -- kept out of the parallel pool as
// ordinary defense-in-depth alongside TextTests/TextMatchTests/RunAnalyzerTests.
[CollectionDefinition(nameof(RowResultsTests), DisableParallelization = true)]
public sealed class RowResultsTestsCollection;

[Collection(nameof(RowResultsTests))]
public class RowResultsTests(SearchQueryTests.QueryCollectionFixture queryCollectionFixture)
    : IClassFixture<SearchQueryTests.QueryCollectionFixture>
{
    private MilvusCollection Collection => queryCollectionFixture.Collection;

    [Fact]
    public async Task QueryRowsAsync_returns_one_dictionary_per_row()
    {
        var rows = await Collection.QueryRowsAsync(
            "id in [2, 3]",
            new QueryParameters { OutputFields = { "varchar", "float_vector" }, ConsistencyLevel = ConsistencyLevel.Strong },
            TestContext.Current.CancellationToken);

        Assert.Equal(2, rows.Count);

        var row2 = Assert.Single(rows, r => (long)r["id"]! == 2);
        Assert.Equal("two", row2["varchar"]);
        Assert.Equal(3.5f, ((ReadOnlyMemory<float>)row2["float_vector"]!).Span[0]);

        var row3 = Assert.Single(rows, r => (long)r["id"]! == 3);
        Assert.Equal("three", row3["varchar"]);
    }

    [Fact]
    public async Task QueryRowsAsync_matches_column_oriented_QueryAsync()
    {
        QueryParameters parameters = new() { OutputFields = { "varchar" }, ConsistencyLevel = ConsistencyLevel.Strong };

        var rows = await Collection.QueryRowsAsync("id in [1, 4]", parameters, TestContext.Current.CancellationToken);
        var columns = await Collection.QueryAsync("id in [1, 4]", parameters, TestContext.Current.CancellationToken);

        var idColumn = (FieldData<long>)Assert.Single(columns, c => c.FieldName == "id");
        var varcharColumn = (FieldData<string?>)Assert.Single(columns, c => c.FieldName == "varchar");

        Assert.Equal(idColumn.Data.Count, rows.Count);
        for (int i = 0; i < idColumn.Data.Count; i++)
        {
            var row = Assert.Single(rows, r => (long)r["id"]! == idColumn.Data[i]);
            Assert.Equal(varcharColumn.Data[i], row["varchar"]);
        }
    }

    [Fact]
    public async Task SearchResults_GetHits_combines_id_score_and_fields()
    {
        SearchResults results = await Collection.SearchAsync(
            "float_vector",
            new ReadOnlyMemory<float>[] { new[] { 3.5f, 4.5f } },
            SimilarityMetricType.L2,
            limit: 2,
            parameters: new SearchParameters { OutputFields = { "varchar" }, ConsistencyLevel = ConsistencyLevel.Strong },
            cancellationToken: TestContext.Current.CancellationToken);

        IReadOnlyList<SearchHit> hits = results.GetHits();

        Assert.Equal(2, hits.Count);

        SearchHit closest = hits[0];
        Assert.Equal(2L, closest.Id);
        Assert.Equal(0f, closest.Score);
        Assert.Equal("two", closest.Fields["varchar"]);
    }

    [Fact]
    public async Task SearchResults_GetHits_slices_by_query_for_multi_vector_search()
    {
        SearchResults results = await Collection.SearchAsync(
            "float_vector",
            new ReadOnlyMemory<float>[] { new[] { 3.5f, 4.5f }, new[] { 9f, 10f } },
            SimilarityMetricType.L2,
            limit: 1,
            parameters: new SearchParameters { OutputFields = { "varchar" }, ConsistencyLevel = ConsistencyLevel.Strong },
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(2, results.NumQueries);

        SearchHit firstQueryHit = Assert.Single(results.GetHits(0));
        Assert.Equal(2L, firstQueryHit.Id);
        Assert.Equal("two", firstQueryHit.Fields["varchar"]);

        SearchHit secondQueryHit = Assert.Single(results.GetHits(1));
        Assert.Equal(5L, secondQueryHit.Id);
        Assert.Equal("five", secondQueryHit.Fields["varchar"]);
    }

    [Fact]
    public async Task SearchResults_GetHits_rejects_out_of_range_queryIndex()
    {
        SearchResults results = await Collection.SearchAsync(
            "float_vector",
            new ReadOnlyMemory<float>[] { new[] { 3.5f, 4.5f } },
            SimilarityMetricType.L2,
            limit: 1,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Throws<ArgumentOutOfRangeException>(() => results.GetHits(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => results.GetHits(-1));
    }

    [Fact]
    public async Task SearchResults_GetHits_returns_empty_fields_without_OutputFields()
    {
        SearchResults results = await Collection.SearchAsync(
            "float_vector",
            new ReadOnlyMemory<float>[] { new[] { 3.5f, 4.5f } },
            SimilarityMetricType.L2,
            limit: 1,
            cancellationToken: TestContext.Current.CancellationToken);

        SearchHit hit = Assert.Single(results.GetHits());
        Assert.Equal(2L, hit.Id);
        Assert.Empty(hit.Fields);
    }
}
