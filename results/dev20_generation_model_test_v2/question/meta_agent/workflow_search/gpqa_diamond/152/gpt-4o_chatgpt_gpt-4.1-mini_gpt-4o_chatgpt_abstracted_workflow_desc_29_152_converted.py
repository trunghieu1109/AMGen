async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction = "Sub-task 1: Extract and summarize the key information about the Michael addition reaction, including reactants, conditions, and the general mechanism."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing reaction, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction = "Sub-task 2: Analyze the relationships between the reactants, conditions, and potential products to understand the reaction mechanism."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing relationships, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2)
        possible_thinkings.append(thinking2)
    final_instr = "Given all the above thinking and answers, find the most consistent and correct solutions for the reaction mechanism"
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, 
                                                 "Sub-task 2: Synthesize and choose the most consistent answer for the reaction mechanism" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    debate_instruction_3 = "Sub-task 3: Explicitly redraw or list the product skeleton with numbered carbons for each reaction. Map substituents onto those labels before writing the IUPAC name."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], 
                                           debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping product skeleton, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], 
                                                 "Sub-task 3: Map product skeleton and provide IUPAC name" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    debate_instruction_4 = "Sub-task 4: Perform a detailed mechanistic analysis of tautomeric equilibria and regioselectivity for each reaction."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], 
                                           debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing mechanism, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Analyze mechanism and provide final answer" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Evaluate the predicted products against the given choices to identify the correct answer."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating products, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Evaluate products and provide final answer" + final_instr, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
