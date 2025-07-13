async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the key components and relationships in the original nuclear decay process (2A -> 2B + 2E + 2V) and the variant (2A -> 2B + 2E + M)."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing decay processes, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_0_2 = "Sub-task 2: Identify and describe the characteristics of the energy spectrum of the outgoing E particles in the original decay process."
    N_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of stage_0_subtask_1", "answer of stage_0_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing energy spectrum, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the energy spectrum."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                             model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2, 
                                                     "Sub-task 2: Synthesize and choose the most consistent answer for energy spectrum" + final_instr_0_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])
    
    debate_instr_1_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_1_1 = "Sub-task 1: Assess the impact of replacing 2V with a massless particle M on the energy conservation and distribution in the decay process." + debate_instr_1_1
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                       model=self.node_model, role=role, temperature=0.5) 
                          for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", "thinking of stage_0_subtask_1", "answer of stage_0_subtask_1", "thinking of stage_0_subtask_2", "answer of stage_0_subtask_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], 
                                               debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assessing impact, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_instr_1_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                             model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1], 
                                                     "Sub-task 1: Assess impact of massless particle M" + final_instr_1_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_1_2 = "Sub-task 2: Transform the understanding of the energy spectrum based on the introduction of the massless particle M and validate the changes against known conservation laws."
    N_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, transforming energy spectrum understanding, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_instr_1_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the transformed energy spectrum."
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                             model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_1_1, answer_1_1] + possible_thinkings_1_2 + possible_answers_1_2, 
                                                     "Sub-task 2: Synthesize and choose the most consistent answer for transformed energy spectrum" + final_instr_1_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {
        "thinking": thinking_1_2,
        "answer": answer_1_2
    }
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instr_2_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_1 = "Sub-task 1: Analyze the adjusted energy spectrum of the outgoing E particles in the variant decay process and classify its characteristics (continuity, endpoint)." + debate_instr_2_1
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                       model=self.node_model, role=role, temperature=0.5) 
                          for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", "thinking of stage_1_subtask_2", "answer of stage_1_subtask_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_2, answer_1_2], 
                                               debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing adjusted energy spectrum, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_instr_2_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                             model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1], 
                                                     "Sub-task 1: Analyze adjusted energy spectrum" + final_instr_2_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])
    
    cot_sc_instruction_2_2 = "Sub-task 2: Compare the adjusted energy spectrum with the original to determine how it changes (discrete vs. continuous, endpoint increase or decrease)."
    N_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of stage_2_subtask_1", "answer of stage_2_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2_2):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, comparing energy spectra, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the comparison of energy spectra."
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                             model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo, thinking_2_1, answer_2_1] + possible_thinkings_2_2 + possible_answers_2_2, 
                                                     "Sub-task 2: Synthesize and choose the most consistent answer for energy spectra comparison" + final_instr_2_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {
        "thinking": thinking_2_2,
        "answer": answer_2_2
    }
    logs.append(subtask_desc_2_2)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
