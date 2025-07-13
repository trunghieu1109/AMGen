async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 0: Analyze the given geometric configuration to extract and compute key parameters. "
        "This includes: (a) formalizing the properties of the convex equilateral hexagon with parallel opposite sides, "
        "(b) representing the hexagon using vectors or coordinates to capture side lengths and directions, "
        "(c) characterizing the triangle formed by the extensions of sides AB, CD, and EF, and "
        "(d) expressing the triangle side lengths (200, 240, 300) in terms of the hexagon's side length and vector directions. "
        "Avoid assuming any specific orientation without justification; instead, use general vector or coordinate geometry principles to maintain generality."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing geometric configuration, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 1: Aggregate the computed parameters and relationships from subtask_0 to set up equations relating the hexagon's side length to the triangle's side lengths. "
        "This involves: (a) combining vector expressions for the hexagon sides and the triangle sides, "
        "(b) selecting appropriate geometric constraints (parallelism, equal side lengths, convexity) to reduce variables, "
        "(c) verifying the consistency of these constraints, and "
        "(d) deriving a solvable system of equations or a formal representation that links the hexagon side length to the given triangle side lengths. "
        "Avoid introducing extraneous assumptions or ignoring the convexity and parallelism conditions."
    )
    N = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, aggregating and setting up equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo] + possible_answers_1 + possible_thinkings_1,
        "Sub-task 1: Synthesize and choose the most consistent answer for the equation setup relating hexagon side length to triangle side lengths.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    cot_instruction_2 = (
        "Sub-task 2: Solve the derived system of equations or geometric relations to find the exact side length of the hexagon. "
        "This includes: (a) performing algebraic manipulations or geometric constructions as needed, "
        "(b) validating the solution against all given conditions (equilateral, convexity, parallel opposite sides, triangle side lengths), and "
        "(c) providing the final numeric value of the hexagon side length. "
        "Avoid accepting extraneous or inconsistent solutions; ensure the solution is physically and geometrically valid."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, solving system and finding hexagon side length, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)

    reflect_inst_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3 = (
        "Sub-task 3: Verify the final solution by cross-checking the derived hexagon side length with the original problem conditions. "
        "This includes: (a) confirming that the hexagon with the found side length satisfies the parallelism and convexity conditions, "
        "(b) ensuring the triangle formed by the extensions of AB, CD, and EF indeed has side lengths 200, 240, and 300, and "
        "(c) providing a concise summary of the verification and final answer. "
        "Avoid skipping any verification step to ensure robustness of the solution. " + reflect_inst_3
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking_0.content, answer_0.content, thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying solution, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3],
                                                   "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback_3.content}; answer: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining solution, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
