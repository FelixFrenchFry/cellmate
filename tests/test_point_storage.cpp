#include <cellmate/neighbor_search.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// verifies that invalid point access modes are rejected
TEST(NeighborSearchPointStorage, RejectsInvalidMode)
{
	EXPECT_THROW(cellmate::NeighborSearch(1.0f, static_cast<cellmate::PointAccess>(-1)), std::invalid_argument);
}

// verifies that default position storage survives input changes and destruction
// (checks both rebuild execution modes)
TEST(NeighborSearchPointStorage, DefaultCopiesPositions)
{
	for (cellmate::ComputeMode execution : { cellmate::ComputeMode::seq, cellmate::ComputeMode::par })
	{
		cellmate::NeighborSearch search(1.0f);
		std::vector<std::size_t> neighbors;
		{
			float x[]{ 8.0f, 0.0f, 4.0f, 0.5f };
			float y[]{ 0.0f, 0.0f, 0.0f, 0.0f };
			float z[]{ 0.0f, 0.0f, 0.0f, 0.0f };
			search.rebuild(x, y, z, 4, execution);
			std::fill(x, x + 4, 100.0f);
			search.query({ 0.0f, 0.0f, 0.0f }, neighbors);
			std::sort(neighbors.begin(), neighbors.end());
			ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 1, 3 }));
		}

		search.query({ 4.0f, 0.0f, 0.0f }, neighbors);
		ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 2 }));
	}
}

// verifies that copied and moved searches keep their query results
// (rebuilds the original search with different points)
TEST(NeighborSearchPointStorage, CopiesAndMovesKeepTheStorageModeAndPositions)
{
	const float x[]{ 8.0f, 0.0f, 4.0f, 0.5f };
	const float other_x[]{ 16.0f, 24.0f, 20.0f, 16.5f };
	const float y[4]{}, z[4]{};
	for (cellmate::PointAccess storage : { cellmate::PointAccess::cpy, cellmate::PointAccess::ref })
	{
		cellmate::NeighborSearch search(1.0f, storage);
		search.rebuild(x, y, z, 4);
		cellmate::NeighborSearch copy = search;
		const auto other_storage = storage == cellmate::PointAccess::cpy
			? cellmate::PointAccess::ref : cellmate::PointAccess::cpy;
		cellmate::NeighborSearch assigned(1.0f, other_storage);
		assigned = search;
		cellmate::NeighborSearch moved = std::move(copy);
		cellmate::NeighborSearch move_assigned(1.0f, other_storage);
		move_assigned = std::move(assigned);
		search.rebuild(other_x, y, z, 4);

		std::vector<std::size_t> neighbors;
		for (const cellmate::NeighborSearch* index : { &moved, &move_assigned })
		{
			index->query({ 0.0f, 0.0f, 0.0f }, neighbors);
			std::sort(neighbors.begin(), neighbors.end());
			ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 1, 3 }));
		}
		search.query({ 16.0f, 0.0f, 0.0f }, neighbors);
		std::sort(neighbors.begin(), neighbors.end());
		ASSERT_EQ(neighbors, (std::vector<std::size_t>{ 0, 3 }));
	}
}
