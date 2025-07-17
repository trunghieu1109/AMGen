async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the given information from the Lagrangian, including definitions and key entities."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0/subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing Lagrangian, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_0_2 = "Sub-task 2: Transform the problem by identifying the mass dimensions of the fields and constants in the Lagrangian."
    N_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0/subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of stage_0/subtask_1", "answer of stage_0/subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, identifying mass dimensions, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the mass dimension problem."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2, 
                                                 "Sub-task 2: Synthesize and choose the most consistent answer for mass dimension" + final_instr_0_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])
    
    reflect_inst_1_1 =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = "Sub-task 3: Assess the impact of the identified mass dimensions on the coupling constant kappa." + reflect_inst_1_1
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1/subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", "thinking of stage_0/subtask_1", "answer of stage_0/subtask_1", "thinking of stage_0/subtask_2", "answer of stage_0/subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, assessing impact on kappa, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    critic_inst_1_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_1):
        feedback_1_1, correct_1_1 = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst_1_1, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_1.content}; answer: {correct_1_1.content}")
        if correct_1_1.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, answer_1_1, feedback_1_1])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining impact assessment, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instr_2_1 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_2_1 = "Sub-task 4: Evaluate the renormalizability of the theory based on the mass dimension of kappa and the structure of the Lagrangian." + debate_instr_2_1
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2/subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", "thinking of stage_1/subtask_1", "answer of stage_1/subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_1, answer_1_1], 
                                           debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating renormalizability, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_instr_2_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[-1] + all_answer_2_1[-1], 
                                                 "Sub-task 4: Evaluate renormalizability" + final_instr_2_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])
    
    cot_sc_instruction_2_2 = "Sub-task 5: Select the correct choice from the given options based on the evaluation of mass dimension and renormalizability."
    N_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2/subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of stage_2/subtask_1", "answer of stage_2/subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2_2):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, selecting correct choice, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the final choice."
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo, thinking_2_1, answer_2_1] + possible_thinkings_2_2 + possible_answers_2_2, 
                                                 "Sub-task 5: Synthesize and choose the most consistent answer for final choice" + final_instr_2_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {
        "thinking": thinking_2_2,
        "answer": answer_2_2
    }
    logs.append(subtask_desc_2_2)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
