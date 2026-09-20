#include <cellmate/neighbor_search.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
	// coordinates of six points
	float x[]{ 0.0f, 0.5f, 2.0f, 0.0f, 0.0f, -0.5f };
	float y[]{ 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,  0.0f };
	float z[]{ 0.0f, 0.0f, 0.0f, 0.0f, 1.5f,  0.0f };

	cellmate::NeighborSearch search(1.0f);
	search.rebuild(x, y, z, 6);

	// find points within radius 1.0 of the origin
	std::vector<std::size_t> neighbors;
	search.find_neighbors({ 0.0f, 0.0f, 0.0f }, neighbors);

	std::cout << "Expected neighbor indices: 0 1 3 5\n";
	std::cout << "Actual neighbor indices: ";
	for (std::size_t index : neighbors)
	{
		std::cout << index << ' ';
	}
	std::cout << '\n';
}
