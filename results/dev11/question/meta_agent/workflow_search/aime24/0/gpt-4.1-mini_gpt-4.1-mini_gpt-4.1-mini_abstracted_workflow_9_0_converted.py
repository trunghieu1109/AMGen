async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Extract and formalize all given information into mathematical expressions. "
        "Define variables s (speed in km/h) and t (coffee time in hours). Convert all times to hours. "
        "Write two key equations representing total time at speeds s and s+2 km/h, including coffee time t. "
        "Emphasize that t is constant and distance is fixed at 9 km. Avoid mixing units and ensure clarity."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and formalizing info, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking_0,
        "answer": answer_0
    }
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 1: Solve the system of two equations derived in Sub-task 0 to find numerical values of s (km/h) and t (hours). "
        "Handle algebraic manipulation carefully and ensure solutions are physically meaningful (positive). "
        "Avoid approximation errors until final calculation."
    )
    N_sc = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_sc_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, solving equations, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo] + possible_answers_1 + possible_thinkings_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct solutions for s and t.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking_1,
        "answer": answer_1
    }
    logs.append(subtask_desc_1)

    debate_instruction_2 = (
        "Sub-task 2: Simplify the expressions for s and t obtained in Sub-task 1 into decimal or fractional forms suitable for further calculations. "
        "Convert t into minutes if necessary. Summarize numeric relationships clearly. "
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate | Reflexion"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_1, answer_1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_1, answer_1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying s and t, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2(
        [taskInfo] + all_thinking_2[-1] + all_answer_2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final simplified numeric answer for s and t.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking_2,
        "answer": answer_2
    }
    logs.append(subtask_desc_2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Calculate the total time (walking plus coffee) when Aya walks at speed s + 0.5 km/h using values of s and t from Sub-task 2. "
        "Convert total time into minutes as requested. Verify result's consistency with given conditions and provide final answer with units."
    )
    N_sc_3 = self.max_sc
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_sc_agents_3[i]([taskInfo, thinking_2, answer_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, calculating total time at s+0.5, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3(
        [taskInfo] + possible_answers_3 + possible_thinkings_3,
        "Sub-task 3: Synthesize and choose the most consistent and correct total time for speed s + 0.5 km/h.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {
        "thinking": thinking_3,
        "answer": answer_3
    }
    logs.append(subtask_desc_3)

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    for i, st in enumerate(sub_tasks):
        print(f"Step {i}: ", st)
    return final_answer, logs
