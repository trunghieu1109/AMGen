async def forward_191(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Assess the electrostatic effect of placing a charge +q inside the cavity of the uncharged spherical conductor, focusing on induced charges and shielding effects on the conductor's surface and cavity boundary. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing electrostatic effects, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_0[r].append(thinking0)
            all_answer_0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_0 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], "stage_0.subtask_1: Electrostatic effect assessment." + final_instr_0, is_sub_task=True)
    agents.append(f"Final Decision agent, stage_0.subtask_1, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"stage_0.subtask_1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1 = "Sub-task 1: Combine and transform the given geometric parameters (R, r, s, L, l, theta) and charge placement information to express the relative positions and distances relevant for calculating the electric field at point P."
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, analyzing geometric relations, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent and correct geometric relations for the problem."
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings_1 + possible_answers_1, "stage_1.subtask_1: Geometric relations." + final_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent, stage_1.subtask_1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"stage_1.subtask_1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Apply electrostatic principles and the method of images or equivalent boundary conditions to relate the induced charges and the original charge +q to the net field outside the conductor, incorporating the displacement s and angle theta."
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking0, answer0, thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking0, answer0, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, applying electrostatic principles, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct electrostatic field expression for the problem."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking0, answer0, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "stage_1.subtask_2: Electrostatic field calculation." + final_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent, stage_1.subtask_2, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"stage_1.subtask_2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instr_2 = "Sub-task 1: Analyze and classify the resulting expressions for the electric field at point P, comparing them with the given choices to identify the correct formula for the magnitude of the electric field considering the geometry and shielding effects. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking3, answer3 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing and classifying expressions, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_2[r].append(thinking3)
            all_answer_2[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_2[-1] + all_answer_2[-1], "stage_2.subtask_1: Final classification and choice." + final_instr_3, is_sub_task=True)
    agents.append(f"Final Decision agent, stage_2.subtask_1, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"stage_2.subtask_1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
