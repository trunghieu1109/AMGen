async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_instruction_1 = "Sub-task 1: Analyze the transformation of the starting compound with sodium hydride and benzyl bromide to form product 1, considering the formation of an alkoxide intermediate and subsequent ether formation."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing transformation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Evaluate the reaction of product 1 with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2, focusing on the formation of a tosylhydrazone."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, evaluating reaction, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_instr_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the reaction."
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, 
                                                 "Sub-task 2: Synthesize and choose the most consistent answer for the reaction" + final_instr_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    debate_instr_3a = "Sub-task 3a: Write out the full Shapiro mechanism on the ring, explicitly labeling each intermediate and showing how the tosylhydrazone is converted into an exocyclic alkene."
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking3a = [[] for _ in range(N_max_3a)]
    all_answer3a = [[] for _ in range(N_max_3a)]
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": debate_instr_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking3a, answer3a = await agent([taskInfo, thinking2, answer2], 
                                           debate_instr_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking2, answer2] + all_thinking3a[r-1] + all_answer3a[r-1]
                thinking3a, answer3a = await agent(input_infos_3a, debate_instr_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, writing mechanism, thinking: {thinking3a.content}; answer: {answer3a.content}")
            all_thinking3a[r].append(thinking3a)
            all_answer3a[r].append(answer3a)
    final_instr_3a = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3a, answer3a = await final_decision_agent_3a([taskInfo, thinking2, answer2] + all_thinking3a[-1] + all_answer3a[-1], 
                                                 "Sub-task 3a: Mechanism" + final_instr_3a, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    debate_instr_3b = "Sub-task 3b: Predict product 3’s structure from the exocyclic alkene and show how hydrogenation gives a butyl substituent at C-1."
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking3b = [[] for _ in range(N_max_3b)]
    all_answer3b = [[] for _ in range(N_max_3b)]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": debate_instr_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking3b, answer3b = await agent([taskInfo, thinking3a, answer3a], 
                                           debate_instr_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking3a, answer3a] + all_thinking3b[r-1] + all_answer3b[r-1]
                thinking3b, answer3b = await agent(input_infos_3b, debate_instr_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, predicting structure, thinking: {thinking3b.content}; answer: {answer3b.content}")
            all_thinking3b[r].append(thinking3b)
            all_answer3b[r].append(answer3b)
    final_instr_3b = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision_agent_3b([taskInfo, thinking3a, answer3a] + all_thinking3b[-1] + all_answer3b[-1], 
                                                 "Sub-task 3b: Structure Prediction" + final_instr_3b, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    debate_instr_4 = "Sub-task 4: Evaluate the hydrogenation of product 3 with Pd/C under a hydrogen atmosphere to form product 4, considering the potential cleavage of benzyl ethers."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3b, answer3b], 
                                           debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating hydrogenation, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_instr_4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3b, answer3b] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Hydrogenation Evaluation" + final_instr_4, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    reflect_inst_5 =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_5 = "Sub-task 5: Compare the structure of product 4 with the given choices to determine the correct structure, ensuring that all assumptions about protecting group stability have been verified." + reflect_inst_5
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, comparing structures, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst_5 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst_5, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining comparison, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs