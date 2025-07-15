async def forward_173(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_sc_instruction_0_1 = "Determine the rest-masses of the two fragments based on the given mass ratio and the 99% mass condition."
    N_0_1 = self.max_sc
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_0_1)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_0_1):
        thinking_0_1, answer_0_1 = await cot_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, determining rest-masses, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
        possible_answers_0_1.append(answer_0_1)
        possible_thinkings_0_1.append(thinking_0_1)
    final_instr_0_1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the rest-masses"
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1 + possible_answers_0_1, 
                                                 "Sub-task 0_1: Synthesize and choose the most consistent answer for rest-masses" + final_instr_0_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_1_1 = "Calculate the total kinetic energy released during the fission process using the mass-energy equivalence principle."
    N_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, calculating total kinetic energy, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_instr_1_1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the total kinetic energy"
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_1_1 + possible_answers_1_1, 
                                                 "Sub-task 1_1: Synthesize and choose the most consistent answer for total kinetic energy" + final_instr_1_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])
    
    debate_instr_1_2 = "Compute the relativistic kinetic energy (T1) of the more massive fragment."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": debate_instr_1_2,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_0_1, answer_0_1, thinking_1_1, answer_1_1], 
                                           debate_instr_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_0_1, answer_0_1, thinking_1_1, answer_1_1] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instr_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing relativistic kinetic energy, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_instr_1_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer for relativistic kinetic energy."
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0_1, answer_0_1, thinking_1_1, answer_1_1] + all_thinking_1_2[-1] + all_answer_1_2[-1], 
                                                 "Sub-task 1_2: Synthesize and choose the most consistent answer for relativistic kinetic energy" + final_instr_1_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {
        "thinking": thinking_1_2,
        "answer": answer_1_2
    }
    logs.append(subtask_desc_1_2)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_1_3 = "Compute the classical kinetic energy (T1) of the more massive fragment."
    N_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_3):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_0_1, answer_0_1, thinking_1_1, answer_1_1], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, computing classical kinetic energy, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_instr_1_3 = "Given all the above thinking and answers, find the most consistent and correct solutions for the classical kinetic energy"
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo, thinking_0_1, answer_0_1, thinking_1_1, answer_1_1] + possible_thinkings_1_3 + possible_answers_1_3, 
                                                 "Sub-task 1_3: Synthesize and choose the most consistent answer for classical kinetic energy" + final_instr_1_3, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {
        "thinking": thinking_1_3,
        "answer": answer_1_3
    }
    logs.append(subtask_desc_1_3)
    print("Step 4: ", sub_tasks[-1])
    
    reflect_inst_2_1 =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Determine the difference between the relativistic and classical kinetic energy values for the more massive fragment." + reflect_inst_2_1
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_2, answer_1_2, thinking_1_3, answer_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_2", "answer of subtask 1_2", "thinking of subtask 1_3", "answer of subtask 1_3"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, determining difference in kinetic energy, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct"
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst_2_1, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining difference in kinetic energy, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
