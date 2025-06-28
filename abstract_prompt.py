USER_PROMPT = """
I want to generalize a set of subtasks extracted from a specific problem-solving scenario so that they can be reused across different domains — not just in mathematics or physics, but also in chemistry, biology, and even social sciences.

The generalization should:

- Avoid referring to any specific numerical data or domain-specific context.
- Focus on the logical role or reasoning pattern each subtask performs.
- Clearly distinguish the purpose and function of each subtask in the overall task-solving workflow.
- Be abstract and high-level enough to apply to various types of queries across disciplines.
- Optionally, help define a reusable subtask template or framework for automating the decomposition of complex problems into such generalized subtasks.

Here is subtask list:
[
    {
        "subtask_id": "subtask_1",
        "objective": "Generate all possible triples (a, b, c) such that a + b + c = 300, allowing for foundational enumeration of candidates.",
        "supporting_info": "This subtask assumes that the integers a, b, and c can take on values from 0 to 300, and the total sum must exactly equal 300.",
        "agent_collaboration": "CoT",
        "dependencies": []
    },
    {
        "subtask_id": "subtask_2",
        "objective": "Explore specific cases such as a = b, a = c, or b = c to simplify the polynomial equation and reduce complexity in further evaluations.",
        "supporting_info": "This subtask builds on the output of subtask 1, focusing on symmetry and cases that might yield simpler forms of the polynomial to facilitate easier filtering of valid triples.",
        "agent_collaboration": "CoT",
        "dependencies": [
            "subtask_1"
        ]
    },
    {
        "subtask_id": "subtask_3",
        "objective": "Utilize identified patterns from subtask 2 to filter triples and ascertain which satisfy the equation a^2b + a^2c + b^2a + b^2c + c^2a + c^2b = 6,000,000.",
        "supporting_info": "This subtask relies heavily on understanding the patterns developed in subtask 2, and iteratively refines potential answers based on feedback to ensure correctness.",
        "agent_collaboration": "CoT (and Reflexion through feedback loops with Critic Agent)",
        "dependencies": [
            "subtask_2"
        ]
    }
]
"""