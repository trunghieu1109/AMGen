async def forward_164(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = "Sub-task 1: Analyze and classify the given information and the four statements regarding the dual catalyst system for ethylene polymerization, focusing on catalyst types, activators, industrial implementation, and economic factors."
    N_sc = self.max_sc
    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, analyze and classify info, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent classification and analysis of the catalyst system and statements.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_1 = "Sub-task 2: Assess the chemical and practical impact of using a second catalyst system with specific activators (including aluminum-based and group VIa transition metal catalysts) on the polymerization process and polymer branching. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_round_1)]
    all_answer_1 = [[] for _ in range(N_round_1)]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking0, answer0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing catalyst impact, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_2 = "Sub-task 3: Derive which of the four statements is chemically and industrially correct based on the analysis of catalyst systems, activator compatibility, and economic considerations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_round_2)]
    all_answer_2 = [[] for _ in range(N_round_2)]
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving correct statement, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking_2[-1] + all_answer_2[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_3 = "Sub-task 4: Combine the derived conclusions to identify and confirm the single correct statement regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system."
    N_sc_3 = self.max_sc
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking3, answer3 = await cot_sc_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, combine conclusions, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings_3 + possible_answers_3, "Sub-task 4: Synthesize and confirm the single correct statement.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
