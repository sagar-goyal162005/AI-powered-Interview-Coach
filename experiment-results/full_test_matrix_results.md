# Complete 40-Run Prompt Engineering Test Matrix

**Date:** 2026-11-17 18:09:29
**Evaluation Model:** Google Gemini (OpenAI Compatible Endpoint: `gemini-3-flash-preview`)
**Total Execution Runs:** 40 (10 Test Cases x 4 Prompt Versions)

## Summary Table

| Run | TC ID | Type | Version | Score | Tech /4 | Comp /2 | Rel /2 | Comm /2 | Source | Model |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | TC01 | Standard Technical Comparison | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 2 | TC01 | Standard Technical Comparison | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 3 | TC01 | Standard Technical Comparison | V3 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 4 | TC01 | Standard Technical Comparison | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 5 | TC02 | Algorithm & Complexity Analysis | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 6 | TC02 | Algorithm & Complexity Analysis | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 7 | TC02 | Algorithm & Complexity Analysis | V3 | **10.0** | 4.0 | 2.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 8 | TC02 | Algorithm & Complexity Analysis | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 9 | TC03 | Pointer Technique & Space Optimization | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 10 | TC03 | Pointer Technique & Space Optimization | V2 | **10.0** | 4.0 | 2.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 11 | TC03 | Pointer Technique & Space Optimization | V3 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 12 | TC03 | Pointer Technique & Space Optimization | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 13 | TC04 | Database Design & Redundancy Reduction | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 14 | TC04 | Database Design & Redundancy Reduction | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 15 | TC04 | Database Design & Redundancy Reduction | V3 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 16 | TC04 | Database Design & Redundancy Reduction | FINAL | **8.0** | 4.0 | 1.0 | 2.0 | 1.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 17 | TC05 | Object-Oriented Design Principles | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 18 | TC05 | Object-Oriented Design Principles | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 19 | TC05 | Object-Oriented Design Principles | V3 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 20 | TC05 | Object-Oriented Design Principles | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 21 | TC06 | OS Concurrency & Memory Isolation | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 22 | TC06 | OS Concurrency & Memory Isolation | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 23 | TC06 | OS Concurrency & Memory Isolation | V3 | **10.0** | 4.0 | 2.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 24 | TC06 | OS Concurrency & Memory Isolation | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 25 | TC07 | Transport Layer Protocols & Reliability | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 26 | TC07 | Transport Layer Protocols & Reliability | V2 | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 27 | TC07 | Transport Layer Protocols & Reliability | V3 | **10.0** | 4.0 | 2.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 28 | TC07 | Transport Layer Protocols & Reliability | FINAL | **9.0** | 4.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 29 | TC08 | System Architecture & End-to-End Design | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 30 | TC08 | System Architecture & End-to-End Design | V2 | **7.0** | 3.0 | 1.0 | 2.0 | 1.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 31 | TC08 | System Architecture & End-to-End Design | V3 | **8.0** | 4.0 | 1.0 | 2.0 | 1.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 32 | TC08 | System Architecture & End-to-End Design | FINAL | **8.0** | 4.0 | 1.0 | 2.0 | 1.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 33 | TC09 | Weak / Partially Correct Answer | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 34 | TC09 | Weak / Partially Correct Answer | V2 | **7.0** | 2.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 35 | TC09 | Weak / Partially Correct Answer | V3 | **7.0** | 2.0 | 2.0 | 2.0 | 1.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 36 | TC09 | Weak / Partially Correct Answer | FINAL | **7.0** | 2.0 | 1.0 | 2.0 | 2.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 37 | TC10 | Edge Case (Empty Answer) | V1 | **7.5** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 38 | TC10 | Edge Case (Empty Answer) | V2 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 39 | TC10 | Edge Case (Empty Answer) | V3 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |
| 40 | TC10 | Edge Case (Empty Answer) | FINAL | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | REAL LLM | `gemini-3.1-flash-lite-preview` |

## Execution Statistics

- **Real LLM Successful Runs:** 40 / 40 (100.0%)
- **Simulated Fallback Runs:** 0 / 40 (0.0%)
- **API Provider Errors:** 0 / 40

## Detailed Analysis by Test Case

### TC01: Standard Technical Comparison
- **Question:** What is the difference between an Array and a Linked List?
- **Student Answer:** An array stores elements in contiguous memory locations with O(1) random access by index, but has a fixed size. A linked list stores elements in separate nodes connected by pointers anywhere in memory, allowing dynamic size and O(1) insertions/deletions at known positions, but requires O(n) sequential access traversal.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | :
"Given that arrays have better cache locality and O(1) access, ... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | In what specific scenarios would you choose a `LinkedList` over a... |
| V3 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | Given that arrays have $O(n)$ insertion costs due to shifting ele... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Excellent | Given the trade-offs you just described, in what specific real-wo... |

### TC02: Algorithm & Complexity Analysis
- **Question:** Explain binary search and its time complexity.
- **Student Answer:** Binary search is an efficient search algorithm on sorted arrays. It works by repeatedly dividing the search interval in half: compare the target value to the middle element; if equal, return index; if target is smaller, search left half; otherwise search right half. Best case time complexity is O(1) if target is at the middle. Average and worst-case time complexity is O(log n) because the search space halves each step. Space complexity is O(1) iteratively or O(log n) recursive call stack.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | Binary search is highly efficient for searching, but what happens... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | How would the performance or implementation of binary search chan... |
| V3 | 10.0 | 4.0/2.0/2.0/2.0 | Strong Answer | If you were searching for a target value in a sorted array that c... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Excellent | In a real-world scenario, how would you modify the binary search ... |

### TC03: Pointer Technique & Space Optimization
- **Question:** How would you detect a cycle in a singly linked list?
- **Student Answer:** We can detect a cycle using Floyd's Cycle-Finding Algorithm, also known as the Tortoise and Hare approach. We initialize two pointers: slow moving one step at a time, and fast moving two steps at a time. If there is a cycle, fast will eventually meet slow inside the loop. If fast reaches NULL, there is no cycle. Time complexity is O(n) and auxiliary space complexity is O(1), which is better than using a hash set with O(n) space.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | That is a great explanation. Building on that, if you have confir... |
| V2 | 10.0 | 4.0/2.0/2.0/2.0 | Strong Answer | If you detect that a cycle exists, how would you find the specifi... |
| V3 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | If you were asked to not just detect the cycle, but to find the *... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Excellent | That is a great explanation. Building on that, if you were asked ... |

### TC04: Database Design & Redundancy Reduction
- **Question:** Explain database normalization and its purpose.
- **Student Answer:** Normalization is the process of organizing data in a relational database to minimize data redundancy and prevent insertion, update, and deletion anomalies. It involves decomposing large tables into smaller, well-structured tables linked by foreign keys. Key normal forms are 1NF (atomic values, no repeating groups), 2NF (in 1NF and no partial dependencies on candidate keys), 3NF (in 2NF and no transitive dependencies), and BCNF (every determinant is a candidate key).

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | In a high-traffic backend system, you have a table that is heavil... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | In what specific scenarios would you intentionally choose to deno... |
| V3 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | In what specific scenarios would you intentionally choose to deno... |
| FINAL | 8.0 | 4.0/1.0/2.0/1.0 | Strong | Given that normalization can lead to complex joins, how would you... |

### TC05: Object-Oriented Design Principles
- **Question:** Explain polymorphism with a software engineering example.
- **Student Answer:** Polymorphism means 'many forms' and allows objects of different classes to be treated as objects of a common superclass. There are two types: compile-time (static) polymorphism achieved through method overloading, and runtime (dynamic) polymorphism achieved through method overriding. For example, a base class 'PaymentMethod' has a method 'processPayment()'. Derived classes 'CreditCardPayment' and 'UPIPayment' override 'processPayment()'. At runtime, the caller can execute 'payment.processPayment()' without knowing the concrete class.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | :
"That was a great explanation. Building on your `PaymentMethod`... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | How does polymorphism contribute to the 'Open/Closed Principle' i... |
| V3 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | How does polymorphism relate to the Open/Closed Principle in SOLI... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Strong | In your `PaymentProcessor` example, how would you handle a scenar... |

### TC06: OS Concurrency & Memory Isolation
- **Question:** What is the difference between a Process and a Thread, and how does context switching differ?
- **Student Answer:** A process is an executing instance of a program with its own independent virtual address space, memory, file descriptors, and Process Control Block (PCB). A thread is the smallest unit of execution within a process; multiple threads of the same process share the same code, data, and heap memory, but have their own registers and call stack (Thread Control Block). Context switching between processes is slower because the OS must flush TLB caches and swap page tables. Thread context switching is faster because virtual memory mapping remains unchanged.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | Since you are interviewing for a Systems Engineer role, the inter... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | Given that threads share the same memory space, what specific syn... |
| V3 | 10.0 | 4.0/2.0/2.0/2.0 | Strong Answer | Given that threads share the same memory space, what are the prim... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Excellent | Given that threads share the same address space, what are the pri... |

### TC07: Transport Layer Protocols & Reliability
- **Question:** Explain why TCP is preferred over UDP for file transfer applications like FTP.
- **Student Answer:** TCP is preferred for file transfer because it is a connection-oriented, reliable transport layer protocol. It guarantees ordered delivery, performs 3-way handshakes, computes checksums, retransmits lost packets via ARQ, and performs flow and congestion control (sliding window). In file transfer, even a single corrupt or missing byte corrupts the entire executable or file. In contrast, UDP is connectionless and does not guarantee packet arrival, ordering, or retransmission, making it suitable for streaming or gaming but disastrous for reliable file downloads.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | Since you mentioned that TCP uses flow and congestion control to ... |
| V2 | 9.0 | 4.0/1.0/2.0/2.0 | Strong Answer | If you were tasked with building a real-time file synchronization... |
| V3 | 10.0 | 4.0/2.0/2.0/2.0 | Strong Answer | Since TCP provides reliability at the cost of latency due to its ... |
| FINAL | 9.0 | 4.0/1.0/2.0/2.0 | Excellent | Since you mentioned that TCP uses congestion control to manage ne... |

### TC08: System Architecture & End-to-End Design
- **Question:** Explain the architecture of a student management system.
- **Student Answer:** The student management system follows a 3-tier microservices-ready architecture. The presentation tier is built with React.js communicating via REST APIs. The application tier consists of a Python Flask backend containing controllers, business service layers, and JWT authentication middleware with role-based access control (Admin, Professor, Student). The data tier uses PostgreSQL with indexed tables for students, courses, grades, and attendance, managed via SQLAlchemy ORM. Redis is used for session caching and caching frequently accessed timetable data.

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | :
"During course registration periods, the system experiences a m... |
| V2 | 7.0 | 3.0/1.0/2.0/1.0 | Good Answer | In a student management system, course registration often leads t... |
| V3 | 8.0 | 4.0/1.0/2.0/1.0 | Good Answer | Given that this system handles sensitive student data, how would ... |
| FINAL | 8.0 | 4.0/1.0/2.0/1.0 | Strong | You mentioned the system is 'microservices-ready.' If we were to ... |

### TC09: Weak / Partially Correct Answer
- **Question:** What is the time and space complexity of QuickSort in the best, average, and worst cases?
- **Student Answer:** QuickSort runs in O(n log n) time in the best and average cases. Worst case time is O(n^2) when the pivot chosen is bad, like always picking the first element in an already sorted list. For space complexity, QuickSort is in-place so its auxiliary space is always O(1).

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | Could you explain the memory implications and time complexity of ... |
| V2 | 7.0 | 2.0/1.0/2.0/2.0 | Good Answer | How can you modify the pivot selection strategy to minimize the p... |
| V3 | 7.0 | 2.0/2.0/2.0/1.0 | Good Answer | Given that QuickSort can degrade to $O(n^2)$ time complexity, wha... |
| FINAL | 7.0 | 2.0/1.0/2.0/2.0 | Good | Given that the worst-case time complexity is $O(n^2)$ due to poor... |

### TC10: Edge Case (Empty Answer)
- **Question:** What are ACID properties in database management systems and why are they important?
- **Student Answer:** [BLANK / EMPTY INPUT]

| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |
|:---:|:---:|:---:|:---|:---|
| V1 | 7.5 | 0.0/0.0/0.0/0.0 | Evaluation Complete | Could you explain the memory implications and time complexity of ... |
| V2 | 0.0 | 0.0/0.0/0.0/0.0 | Needs Improvement | Can you explain the trade-offs between ACID and BASE models, and ... |
| V3 | 0.0 | 0.0/0.0/0.0/0.0 | Needs Improvement | *   "Can you explain the trade-off between Isolation levels (e.g.... |
| FINAL | 0.0 | 0.0/0.0/0.0/0.0 | Needs Improvement | In the context of the 'Isolation' property, can you explain what ... |
