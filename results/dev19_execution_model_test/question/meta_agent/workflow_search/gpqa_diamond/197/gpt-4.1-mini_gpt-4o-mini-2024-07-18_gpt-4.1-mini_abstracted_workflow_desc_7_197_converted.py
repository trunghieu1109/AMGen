async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = "Sub-task 1: Extract and summarize all given information from the query, including concentrations, stability constants, and the target complex definition. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, extracting info, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 1: Synthesize extracted information." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_2 = "Sub-task 2: Evaluate and prioritize the significance of the stability constants and concentrations to understand their impact on complex formation equilibria. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instr_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, prioritizing constants, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_2[r].append(thinking2)
            all_answer_2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking_2[-1] + all_answer_2[-1], "Sub-task 2: Synthesize prioritized significance." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = "Sub-task 3.1: Analyze and classify the cobalt(II) species in solution by writing equilibrium expressions for all complexes using the given stability constants and initial concentrations."
    N_sc = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3_1, answer3_1 = await cot_agents_3_1[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, analyzing species, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_answers_3_1.append(answer3_1)
        possible_thinkings_3_1.append(thinking3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings_3_1 + possible_answers_3_1, "Sub-task 3.1: Synthesize equilibrium expressions." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_sc_instruction_3_2 = "Sub-task 3.2: Formulate the mass balance and equilibrium equations to relate free Co(II), free SCN-, and all complex species concentrations."
    cot_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_2 = []
    possible_thinkings_3_2 = []
    subtask_desc3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_3_2,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3_2, answer3_2 = await cot_agents_3_2[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_2[i].id}, formulating equations, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
        possible_answers_3_2.append(answer3_2)
        possible_thinkings_3_2.append(thinking3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_2, answer3_2 = await final_decision_agent_3_2([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings_3_2 + possible_answers_3_2, "Sub-task 3.2: Synthesize mass balance and equilibrium equations." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])

    debate_instr_4 = "Sub-task 4: Solve the system of equilibrium and mass balance equations to determine the concentration of the blue dithiocyanato cobalt(II) complex and total cobalt species in solution. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_4,
        "context": ["user query", thinking3_1, answer3_1, thinking3_2, answer3_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3_1, answer3_1, thinking3_2, answer3_2] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 4: Synthesize solution of system." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_5 = "Sub-task 5: Calculate the percentage of the blue dithiocyanato cobalt(II) complex relative to the total cobalt concentration and select the closest matching choice from the given options. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instr_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instr_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instr_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating percentage, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Final synthesis and selection of closest choice." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
