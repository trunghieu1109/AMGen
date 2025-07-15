async def forward_193(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = "Sub-task 1: Extract and summarize the defining features of the spin system, including spin values, energy formula, and parameters (J, beta)."
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting system features, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1: Analyze and classify all possible spin configurations (8 total) by calculating their energies using the given formula, based on output from stage_0.subtask_1."
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, analyzing spin configurations, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0, answer_0] + possible_thinkings_1_1 + possible_answers_1_1, "Sub-task 1: Synthesize and choose the most consistent classification of spin configurations and their energies.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = "Sub-task 2: Given the classified energies and configurations from stage_1.subtask_1, group configurations by energy levels and determine degeneracies. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, grouping energies and degeneracies, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[-1] + all_answer_1_2[-1], "Sub-task 2: Finalize grouping of energies and degeneracies.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = "Sub-task 1: Compute the Boltzmann weight exp(-beta*E) for each distinct energy level and multiply by its degeneracy, based on output from stage_1.subtask_2."
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_1, answer_2_1 = await cot_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, computing Boltzmann weights, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_2, answer_1_2] + possible_thinkings_2_1 + possible_answers_2_1, "Sub-task 1: Finalize Boltzmann weights for each energy level.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_instruction_3_1 = "Sub-task 1: Sum the weighted contributions from all energy levels to obtain the partition function Z, based on output from stage_2.subtask_1."
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_1, answer_2_1], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, summing weighted contributions, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_4_1 = "Sub-task 1: Evaluate the derived partition function expression against the given choices and select the correct one. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4_1 = self.max_round
    all_thinking_4_1 = [[] for _ in range(N_max_4_1)]
    all_answer_4_1 = [[] for _ in range(N_max_4_1)]
    subtask_desc_4_1 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking_4_1, answer_4_1 = await agent([taskInfo, thinking_3_1, answer_3_1], debate_instruction_4_1, r, is_sub_task=True)
            else:
                input_infos_4_1 = [taskInfo, thinking_3_1, answer_3_1] + all_thinking_4_1[r-1] + all_answer_4_1[r-1]
                thinking_4_1, answer_4_1 = await agent(input_infos_4_1, debate_instruction_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating partition function choices, thinking: {thinking_4_1.content}; answer: {answer_4_1.content}")
            all_thinking_4_1[r].append(thinking_4_1)
            all_answer_4_1[r].append(answer_4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4_1, answer_4_1 = await final_decision_agent_4_1([taskInfo, thinking_3_1, answer_3_1] + all_thinking_4_1[-1] + all_answer_4_1[-1], "Sub-task 1: Final decision on the correct partition function choice.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_4.subtask_1 output: thinking - {thinking_4_1.content}; answer - {answer_4_1.content}")
    subtask_desc_4_1['response'] = {"thinking": thinking_4_1, "answer": answer_4_1}
    logs.append(subtask_desc_4_1)
    print("Step 4.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4_1, answer_4_1, sub_tasks, agents)
    return final_answer, logs
