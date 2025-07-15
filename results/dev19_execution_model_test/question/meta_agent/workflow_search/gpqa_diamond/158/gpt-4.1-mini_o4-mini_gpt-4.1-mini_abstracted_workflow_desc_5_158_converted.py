async def forward_158(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = "Sub-task 1: Extract and summarize the defining observational features of the quasar spectrum and the given cosmological parameters from the user query."
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and summarizing observational and cosmological data, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Analyze the spectral data to infer the quasar's redshift by relating the observed 790 nm peak to a plausible rest-frame wavelength. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent([taskInfo, thinking_0, answer_0], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking_0, answer_0] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, inferring redshift, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0, answer_0] + all_thinking_1_1[-1] + all_answer_1_1[-1], "Sub-task 1: Infer quasar redshift by synthesizing debate outputs. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Classify the cosmological model parameters and their implications for the redshift-distance relation in a flat Lambda-CDM universe, based on the output from Sub-task 1."
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0, answer_0, thinking_1_1, answer_1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, classifying cosmological parameters, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1] + possible_thinkings_1_2 + possible_answers_1_2, "Sub-task 2: Synthesize and choose the most consistent classification of cosmological parameters and implications.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 1: Transform the inferred redshift and cosmological parameters into a quantitative comoving distance estimate using the Lambda-CDM model's distance-redshift relation. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating comoving distance, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 2: Synthesize and provide final comoving distance estimate.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3_1 = "Sub-task 1: Evaluate and prioritize the candidate comoving distance values (6, 7, 8, 9 Gpc) against the calculated estimate to select the most plausible value for the quasar's comoving distance. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_3_1, answer_3_1 = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_1, answer_2_1] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_3_1, answer_3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate distances, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
            all_thinking_3_1[r].append(thinking_3_1)
            all_answer_3_1[r].append(answer_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo, thinking_2_1, answer_2_1] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 3: Select the most plausible comoving distance value.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
