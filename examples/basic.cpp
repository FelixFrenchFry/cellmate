#include <cellmate/neighbor_search.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
	// positions stored in separate 'x', 'y', 'z' arrays
	float x[]{ 0.0f, 0.5f, 2.0f, 0.0f, 0.0f, -0.5f };
	float y[]{ 0.0f, 0.0f, 0.0f, 0.5f, 0.0f,  0.0f };
	float z[]{ 0.0f, 0.0f, 0.0f, 0.0f, 1.5f,  0.0f };

	cellmate::NeighborSearch search(1.0f);
	search.rebuild(x, y, z, 6);

	// find points within radius 1.0 of the origin
	std::vector<std::size_t> neighbors;
	search.query({ 0.0f, 0.0f, 0.0f }, neighbors);

	std::cout << "Expected neighbor set: 0 1 3 5\n";
	std::cout << "Actual neighbor set: ";
	for (std::size_t index : neighbors)
	{
		std::cout << index << ' ';
	}
	std::cout << '\n';
}
