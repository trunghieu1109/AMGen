async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the key components of the original nuclear decay process and the variant process."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing decay process, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    
    debate_instruction_0_2 = "Sub-task 2: Identify and describe the relationships between the components of the decay processes, focusing on energy conservation and particle interactions."
    debate_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) 
                        for role in self.debate_role]
    N_max_0_2 = self.max_round
    all_thinking_0_2 = [[] for _ in range(N_max_0_2)]
    all_answer_0_2 = [[] for _ in range(N_max_0_2)]
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": debate_instruction_0_2,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_2):
        for i, agent in enumerate(debate_agents_0_2):
            if r == 0:
                thinking_0_2, answer_0_2 = await agent([taskInfo, thinking_0_1, answer_0_1], 
                                               debate_instruction_0_2, r, is_sub_task=True)
            else:
                input_infos_0_2 = [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[r-1] + all_answer_0_2[r-1]
                thinking_0_2, answer_0_2 = await agent(input_infos_0_2, debate_instruction_0_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing relationships, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
            all_thinking_0_2[r].append(thinking_0_2)
            all_answer_0_2[r].append(answer_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[-1] + all_answer_0_2[-1], 
                                                     "Sub-task 0_2: Analyze relationships" + final_instr_0_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    
    cot_sc_instruction_1_1 = "Sub-task 1: Assess the impact of replacing 2V with a massless particle M on the energy spectrum of the E particles."
    N_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of subtask 0_2", "answer of subtask 0_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, assessing impact, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_instr_1_1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the impact assessment."
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_2, answer_0_2] + possible_thinkings_1_1 + possible_answers_1_1, 
                                                     "Sub-task 1_1: Synthesize impact assessment" + final_instr_1_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    
    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = "Sub-task 2: Transform the understanding of the decay process to reflect the changes in the energy spectrum due to the introduction of particle M." + reflect_inst_1_2
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                   model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking_0_2, answer_0_2, thinking_1_1, answer_1_1]
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", "thinking of subtask 0_2", "answer of subtask 0_2", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining understanding, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    critic_inst_1_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_2):
        feedback_1_2, correct_1_2 = await critic_agent_1_2([taskInfo, thinking_1_2, answer_1_2], 
                                               "Please review and provide the limitations of provided solutions" + critic_inst_1_2, 
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")
        if correct_1_2.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2, answer_1_2, feedback_1_2])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining understanding, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {
        "thinking": thinking_1_2,
        "answer": answer_1_2
    }
    logs.append(subtask_desc_1_2)
    
    debate_instruction_2_1 = "Sub-task 1: Analyze the adjusted energy spectrum of the E particles and classify it according to the provided choices."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) 
                        for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_2", "answer of subtask 1_2"],
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
            agents.append(f"Debate agent {agent.id}, round {r}, classifying spectrum, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_instr_2_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1], 
                                                     "Sub-task 2_1: Classify spectrum" + final_instr_2_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    
    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs