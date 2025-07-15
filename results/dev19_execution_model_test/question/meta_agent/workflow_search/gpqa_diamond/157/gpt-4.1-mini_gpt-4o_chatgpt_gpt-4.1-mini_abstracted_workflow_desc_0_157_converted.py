async def forward_157(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Analyze and classify the molecular components and mutations described: the transcription factor subunit, phosphorylation sites, transactivation and dimerization domains, and the nature of mutations X and Y (recessive loss-of-function vs dominant-negative). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing molecular components and mutations, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], "Sub-task 1: Synthesize and choose the most consistent classification of molecular components and mutations." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_1 = "Sub-task 1: Assess the molecular impact of mutation Y in the dimerization domain, focusing on how a dominant-negative mutation typically affects protein dimerization and function. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_1, answer_1 = await agent([taskInfo, thinking_0, answer_0], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking_0, answer_0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking_1, answer_1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing molecular impact of mutation Y, thinking: {thinking_1.content}; answer: {answer_1.content}")
            all_thinking_1[r].append(thinking_1)
            all_answer_1[r].append(answer_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo, thinking_0, answer_0] + all_thinking_1[-1] + all_answer_1[-1], "Sub-task 2: Synthesize and choose the most consistent assessment of mutation Y's molecular impact." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 1: Based on the outputs from Sub-task 1 and Sub-task 2, derive the most plausible molecular phenotype caused by mutation Y, considering its dominant-negative effect and the known behavior of transcription factor dimers."
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_0, answer_0, thinking_1, answer_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_0, answer_0, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving plausible molecular phenotype, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_0, answer_0, thinking_1, answer_1] + possible_thinkings_2 + possible_answers_2, "Sub-task 3: Synthesize and choose the most consistent molecular phenotype caused by mutation Y." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 1: Combine the analysis and derived phenotype to select the most likely molecular phenotype from the provided options, justifying the choice based on dominant-negative mutation mechanisms. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2, answer_2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting most likely molecular phenotype, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo, thinking_2, answer_2] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 4: Final selection of the most likely molecular phenotype from the provided options." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
