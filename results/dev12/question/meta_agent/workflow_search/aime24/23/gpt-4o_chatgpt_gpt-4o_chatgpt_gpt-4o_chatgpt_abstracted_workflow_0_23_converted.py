async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    # Stage 1: Constraint Identification and Analysis
    cot_instruction_1_1 = "Sub-task 1: Identify and clearly state the constraints for the row sums and column sums based on the given problem."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1_1, answer1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, analyzing constraints, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {
        "thinking": thinking1_1,
        "answer": answer1_1
    }
    logs.append(subtask_desc1_1)
    print("Step 1: ", sub_tasks[-1])
    
    # Stage 2: Formulation of Equations
    cot_instruction_2_1 = "Sub-task 2: Formulate the problem as a set of equations representing the constraints on the row and column sums."
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1_1, answer1_1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, formulating equations, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {
        "thinking": thinking2_1,
        "answer": answer2_1
    }
    logs.append(subtask_desc2_1)
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 3: Enumeration of Valid Digit Pairs
    cot_sc_instruction_3_1 = "Sub-task 3: Enumerate all valid digit pairs (a,d), (b,e), and (c,f) with digits strictly in the range 0–9, filtering out invalid pairs."
    N = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", "thinking of subtask 2_1", "answer of subtask 2_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3_1, answer3_1 = await cot_agents_3_1[i]([taskInfo, thinking2_1, answer2_1], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, enumerating valid digit pairs, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_answers_3_1.append(answer3_1)
        possible_thinkings_3_1.append(thinking3_1)
    final_instr_3_1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the digit pairs."
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, 
                                                 "Sub-task 3_1: Synthesize and choose the most consistent answer for digit pairs" + final_instr_3_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {
        "thinking": thinking3_1,
        "answer": answer3_1
    }
    logs.append(subtask_desc3_1)
    print("Step 3: ", sub_tasks[-1])
    
    # Stage 4: Systematic Generation and Verification
    debate_instr_4_1 = "Sub-task 4: Systematically generate all possible combinations of valid digit pairs and verify the original equations explicitly."
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_4_1 = self.max_round
    all_thinking4_1 = [[] for _ in range(N_max_4_1)]
    all_answer4_1 = [[] for _ in range(N_max_4_1)]
    subtask_desc4_1 = {
        "subtask_id": "subtask_4_1",
        "instruction": debate_instr_4_1,
        "context": ["user query", "thinking of subtask 3_1", "answer of subtask 3_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking4_1, answer4_1 = await agent([taskInfo, thinking3_1, answer3_1], 
                                           debate_instr_4_1, r, is_sub_task=True)
            else:
                input_infos_4_1 = [taskInfo, thinking3_1, answer3_1] + all_thinking4_1[r-1] + all_answer4_1[r-1]
                thinking4_1, answer4_1 = await agent(input_infos_4_1, debate_instr_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating and verifying combinations, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
            all_thinking4_1[r].append(thinking4_1)
            all_answer4_1[r].append(answer4_1)
    final_instr_4_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4_1, answer4_1 = await final_decision_agent_4_1([taskInfo] + all_thinking4_1[-1] + all_answer4_1[-1], 
                                                 "Sub-task 4_1: Final decision on valid configurations" + final_instr_4_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 4_1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {
        "thinking": thinking4_1,
        "answer": answer4_1
    }
    logs.append(subtask_desc4_1)
    print("Step 4: ", sub_tasks[-1])
    
    # Stage 5: Reflexion and Final Answer
    reflect_inst_5_1 =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_5_1 = "Sub-task 5: Reflexion on the problem." + reflect_inst_5_1
    cot_agent_5_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_5_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_5_1 = self.max_round
    cot_inputs_5_1 = [taskInfo, thinking4_1, answer4_1]
    subtask_desc5_1 = {
        "subtask_id": "subtask_5_1",
        "instruction": cot_reflect_instruction_5_1,
        "context": ["user query", "thinking of subtask 4_1", "answer of subtask 4_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking5_1, answer5_1 = await cot_agent_5_1(cot_inputs_5_1, cot_reflect_instruction_5_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5_1.id}, refining valid scenarios, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    critic_inst_5_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_5_1):
        feedback_5_1, correct_5_1 = await critic_agent_5_1([taskInfo, thinking5_1, answer5_1], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst_5_1, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5_1.id}, providing feedback, thinking: {feedback_5_1.content}; answer: {correct_5_1.content}")
        if correct_5_1.content == "True":
            break
        cot_inputs_5_1.extend([thinking5_1, answer5_1, feedback_5_1])
        thinking5_1, answer5_1 = await cot_agent_5_1(cot_inputs_5_1, cot_reflect_instruction_5_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5_1.id}, refining valid scenarios, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    sub_tasks.append(f"Sub-task 5_1 output: thinking - {thinking5_1.content}; answer - {answer5_1.content}")
    subtask_desc5_1['response'] = {
        "thinking": thinking5_1,
        "answer": answer5_1
    }
    logs.append(subtask_desc5_1)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5_1, answer5_1, sub_tasks, agents)
    return final_answer, logs