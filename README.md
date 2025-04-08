# LeetCode Solutions in Kotlin

This repository contains Kotlin solutions to various LeetCode problems. Each solution is implemented as a unit test for easy execution and verification.

## Problems Solved

The following LeetCode problems are solved in this repository:

| Problem Number | Title                                                      | Difficulty | Time Complexity | Space Complexity |
|----------------|------------------------------------------------------------|------------|-----------------|------------------|
| 02             | Container With Most Water                                  | Medium     | O(n)            | O(1)             |
| 03             | Valid Mountain Array                                       | Easy       | O(n)            | O(1)             |
| 04             | Boats to Save People                                       | Medium     | O(n log n)      | O(1)             |
| 05             | Move Zeroes                                                | Easy       | O(n)            | O(1)             |
| 06             | Longest Substring Without Repeating Characters             | Medium     | O(n)            | O(min(m, n))     |
| 07             | Find First and Last Position of Element in Sorted Array | Medium     | O(log n)        | O(1)             |
| 08             | First Bad Version                                          | Easy       | O(log n)        | O(1)             |
| 09             | Missing Number                                             | Easy       | O(n)            | O(1)             |
| 10             | Count Primes                                               | Medium     | O(n log log n)  | O(n)             |
| 11             | Single Number                                              | Easy       | O(n)            | O(1)             |
| 12             | Robot Return to Origin                                     | Easy       | O(n)            | O(1)             |
| 13             | Add Binary                                                 | Easy       | O(n)            | O(n)             |
| 14             | Two Sum                                                    | Easy       | O(n)            | O(n)             |
| 15             | Contains Duplicate                                         | Easy       | O(n)            | O(n)             |
| 17             | Majority Element                                           | Easy       | O(n)            | O(1)             |
| 18             | Minimum Window Substring                                   | Hard       | O(n)            | O(n)             |
| 19             | Group Anagrams                                             | Medium     | O(nk log k)     | O(nk)            |
| 20             | LRU Cache                                                  | Medium     | O(1)            | O(n)             |
| 32             | Maximum Depth of Binary Tree                               | Easy       | O(n)            | O(h)             |
| 35             | Kth Smallest Element in a BST                              | Medium     | O(n)            | O(h)             |
| 36             | Serialize and Deserialize Binary Tree                      | Hard       | O(n)            | O(n)             |
| 37             | Binary Tree Maximum Path Sum                               | Hard       | O(n)            | O(h)             |
| 38             | Min Stack                                                  | Easy       | O(1)            | O(1)             |
| 43             | House Robber                                               | Easy       | O(n)            | O(1)             |
| 46             | Coin Change                                                | Medium     | O(n)            | O(1)             |

**Note:**
- 'n' represents the input size (e.g., number of elements in an array, length of a string, number of nodes in a tree).
- 'k' in "Group Anagrams" refers to the maximum length of a string in the input array.
- 'h' in tree-related problems represents the height of the binary tree.
- 'm' in "Longest Substring Without Repeating Characters" represents the size of the character set.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```

2.  **Open the project** in your preferred Kotlin/Java IDE (like IntelliJ IDEA).

3.  **Run the unit tests:** You can run each solution individually or all of them together using the IDE's testing framework. The output will show the problem title, the input used for testing, the calculated output, and the time and space complexity of the solution.

## Structure

The code is organized within a single Kotlin class `ExampleUnitTest`. Each LeetCode problem solution is implemented as a private function within this class, and a corresponding unit test function (annotated with `@Test`) is provided to demonstrate its usage and verify its correctness.

## Contributing

Contributions to this repository are welcome. If you have a more efficient solution or want to add solutions to other LeetCode problems, feel free to submit a pull request. Please ensure that your code includes a unit test and follows a similar structure to the existing solutions.

## License

This project is licensed under the [MIT License](LICENSE).
