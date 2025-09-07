#include <iostream>
#include <vector>
using namespace std;

// Function to find the maximum sum of non-adjacent elements
int maxNonAdjacentSum(const vector<int>& nums) {
    int include = 0; // Max sum including the previous element
    int exclude = 0; // Max sum excluding the previous element

    for (int num : nums) {
        int new_exclude = max(include, exclude); // Max sum excluding current element
        include = exclude + num; // Update include to the current number
        exclude = new_exclude; // Update exclude
    }

    return max(include, exclude); // Return the max of include and exclude
}

int main() {
    vector<int> redElements = {2, 4, 6, 2, 5}; // Example input
    cout << "Maximum sum of non-adjacent red elements: " << maxNonAdjacentSum(redElements) << endl;
    return 0;
}