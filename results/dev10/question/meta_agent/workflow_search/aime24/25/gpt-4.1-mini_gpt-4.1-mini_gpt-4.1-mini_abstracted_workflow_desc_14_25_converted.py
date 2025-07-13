async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0 = (
        "Sub-task 1: Identify and verify the geometric elements and constraints of the problem. "
        "Specifically, confirm that ABCDEF is a convex equilateral hexagon with three pairs of opposite sides parallel, "
        "making it a parallelogon. Enumerate the parallel pairs (AB ∥ DE, BC ∥ EF, CD ∥ FA) and establish that all sides have equal length s. "
        "Clarify the definition of the triangle formed by the extensions of AB, CD, and EF as the triangle formed by the intersection points of the lines containing these sides. "
        "Avoid assuming the hexagon is regular (equal angles) unless proven."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, verifying geometric elements, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])
    
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Derive vector or geometric representations for the sides of the hexagon. "
        "Assign vectors to sides AB, BC, CD, DE, EF, and FA such that opposite sides are parallel and equal in length, and all sides have length s. "
        "Express the directions of AB, CD, and EF and their corresponding lines. Determine the equations of the lines containing AB, CD, and EF, preparing to find their intersection points to form the triangle. "
        "Use vector addition and properties of parallelogons. Avoid introducing unnecessary coordinate systems until necessary."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, deriving vector representations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and choose the most consistent vector representation for the hexagon sides.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])
    
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Validate the relationships between the vectors representing the hexagon sides and the triangle formed by the extended lines AB, CD, and EF. "
        "Confirm that the triangle's side lengths (200, 240, 300) correspond to distances between intersection points of these lines. "
        "Formulate expressions for these distances in terms of the hexagon side length s and the directions of the sides. Avoid assumptions about angle sizes without derivation."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0.content, answer_0.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i](
            [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1],
            cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, validating vector-triangle relations, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 2: Synthesize and choose the most consistent validation of vector-triangle side length relations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])
    
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Infer and compute the side length s of the hexagon by relating the given triangle side lengths (200, 240, 300) "
        "to the vector-derived expressions for the triangle sides formed by the extended lines. Use geometric or trigonometric identities and vector equations to solve for s. "
        "Carefully consider the ratios and constraints imposed by the parallelogon structure. Avoid skipping algebraic steps to ensure correctness."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i](
            [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
            cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, computing hexagon side length s, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + possible_answers_2_1 + possible_thinkings_2_1,
        "Sub-task 1: Synthesize and choose the most consistent computed side length s.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])
    
    debate_instruction_3_1 = (
        "Sub-task 1: Decompose, simplify, and sum any components derived in the previous stage to produce the final numeric value of the hexagon's side length s. "
        "Simplify expressions to their minimal form and verify that the result is consistent with all given constraints. "
        "Provide a final answer with verification through back-substitution or geometric reasoning. Avoid leaving the answer in an unsimplified or ambiguous form. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_1, answer_2_1] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, refining final numeric value, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
