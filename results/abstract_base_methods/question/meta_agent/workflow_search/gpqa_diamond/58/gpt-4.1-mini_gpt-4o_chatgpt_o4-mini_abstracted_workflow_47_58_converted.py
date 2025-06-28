async def forward_58(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Calculate total moles and total volume
    cot_agents_stage1 = {}
    subtask_results = {}
    
    # Sub-task 1: Calculate total moles of CH3COOH
    cot_instruction_1 = "Sub-task 1: Calculate the total moles of CH3COOH by multiplying its concentration by its volume in liters, with context from the query."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating total moles of CH3COOH, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    subtask_results['subtask_1'] = answer1.content
    
    # Sub-task 2: Calculate total moles of HCl
    cot_instruction_2 = "Sub-task 2: Calculate the total moles of HCl by multiplying its concentration by its volume in liters, with context from the query."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total moles of HCl, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    subtask_results['subtask_2'] = answer2.content
    
    # Sub-task 3: Calculate total moles of Ba(OH)2
    cot_instruction_3 = "Sub-task 3: Calculate the total moles of Ba(OH)2 by multiplying its concentration by its volume in liters, with context from the query."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating total moles of Ba(OH)2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    subtask_results['subtask_3'] = answer3.content
    
    # Sub-task 4: Calculate total volume of combined solution
    cot_instruction_4 = "Sub-task 4: Calculate the total volume of the combined solution by summing the individual volumes of CH3COOH, HCl, and Ba(OH)2 solutions, with context from the query."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating total volume of combined solution, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    subtask_results['subtask_4'] = answer4.content
    
    # Stage 2: Calculate moles of OH-, net moles of H+, concentration, pH, and select correct choice
    
    # Sub-task 5: Calculate total moles of OH- ions from Ba(OH)2
    cot_instruction_5 = "Sub-task 5: Calculate the total moles of OH- ions contributed by Ba(OH)2, considering each mole produces two moles of OH-, using output from Sub-task 3."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating total moles of OH- ions from Ba(OH)2, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    subtask_results['subtask_5'] = answer5.content
    
    # Sub-task 6: Determine net moles of H+ ions
    cot_instruction_6 = "Sub-task 6: Determine the net moles of H+ ions by subtracting moles of OH- ions from total moles of H+ ions contributed by HCl and partial dissociation of CH3COOH, using outputs from Sub-tasks 1, 2, and 5."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking5.content, answer5.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking1, answer1, thinking2, answer2, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, determining net moles of H+ ions, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    subtask_results['subtask_6'] = answer6.content
    
    # Sub-task 7: Calculate concentration of H+ or OH- ions
    cot_instruction_7 = "Sub-task 7: Calculate the concentration of H+ or OH- ions by dividing net moles by total volume, using outputs from Sub-tasks 4 and 6."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking4.content, answer4.content, thinking6.content, answer6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking4, answer4, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating concentration of H+ or OH- ions, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    subtask_results['subtask_7'] = answer7.content
    
    # Sub-task 8: Calculate pH or pOH and then pH
    cot_instruction_8 = "Sub-task 8: Calculate the pH of the solution using concentration of H+ ions if acidic, or calculate pOH and then pH if basic, applying pH + pOH = 14, using output from Sub-task 7."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, calculating pH of the solution, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    subtask_results['subtask_8'] = answer8.content
    
    # Sub-task 9: Compare calculated pH with given choices and select correct letter
    debate_instruction_9 = "Sub-task 9: Compare the calculated pH value with the given multiple-choice options and select the correct letter choice (A, B, C, or D) corresponding to the closest pH value, using output from Sub-task 8."
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", thinking8.content, answer8.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking8, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct pH choice, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct pH choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on selecting correct pH choice, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
