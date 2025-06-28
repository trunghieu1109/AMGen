async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Define variables a,b,c,d,e,f representing the digits in the 2x3 grid cells consistently, where a,b,c are the top row digits and d,e,f are the bottom row digits." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, define digit variables a,b,c,d,e,f consistently, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Express the two horizontal numbers formed by reading each row left to right as algebraic expressions using variables a,b,c,d,e,f." 
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, express horizontal numbers as algebraic expressions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Set up the equation that the sum of the two horizontal numbers equals 999, using the expressions from subtask 2." 
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, set up horizontal sum equation equals 999, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Express the three vertical numbers formed by reading each column top to bottom as algebraic expressions using variables a,b,c,d,e,f." 
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, express vertical numbers as algebraic expressions, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Derive explicit digit-sum constraints from the horizontal sum equation (from subtask 3), such as a + d = 9, b + e = 9, c + f = 9, by analyzing place-value addition and carry-over conditions." 
    debate_agents_5a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5a = self.max_round
    all_thinking5a = [[] for _ in range(N_max_5a)]
    all_answer5a = [[] for _ in range(N_max_5a)]
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5a):
        for i, agent in enumerate(debate_agents_5a):
            if r == 0:
                thinking5a, answer5a = await agent([taskInfo, thinking3, answer3], cot_instruction_5a, r, is_sub_task=True)
            else:
                input_infos_5a = [taskInfo, thinking3, answer3] + all_thinking5a[r-1] + all_answer5a[r-1]
                thinking5a, answer5a = await agent(input_infos_5a, cot_instruction_5a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, derive digit-sum constraints from horizontal sum, thinking: {thinking5a.content}; answer: {answer5a.content}")
            all_thinking5a[r].append(thinking5a)
            all_answer5a[r].append(answer5a)
    final_decision_agent_5a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5a, answer5a = await final_decision_agent_5a([taskInfo] + all_thinking5a[-1] + all_answer5a[-1], "Sub-task 5a: Make final decision on digit-sum constraints from horizontal sum.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing digit-sum constraints from horizontal sum, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Derive explicit sum constraints from the vertical sum equation (from subtask 5), such as a + b + c = 8 and d + e + f = 19, by expanding and simplifying the equation." 
    debate_agents_5b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5b = self.max_round
    all_thinking5b = [[] for _ in range(N_max_5b)]
    all_answer5b = [[] for _ in range(N_max_5b)]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5b):
        for i, agent in enumerate(debate_agents_5b):
            if r == 0:
                thinking5b, answer5b = await agent([taskInfo, thinking4, answer4], cot_instruction_5b, r, is_sub_task=True)
            else:
                input_infos_5b = [taskInfo, thinking4, answer4] + all_thinking5b[r-1] + all_answer5b[r-1]
                thinking5b, answer5b = await agent(input_infos_5b, cot_instruction_5b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, derive sum constraints from vertical sum, thinking: {thinking5b.content}; answer: {answer5b.content}")
            all_thinking5b[r].append(thinking5b)
            all_answer5b[r].append(answer5b)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5b, answer5b = await final_decision_agent_5b([taskInfo] + all_thinking5b[-1] + all_answer5b[-1], "Sub-task 5b: Make final decision on sum constraints from vertical sum.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing sum constraints from vertical sum, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_6a = "Sub-task 6a: Analyze the combined digit-sum constraints from subtasks 5a and 5b to establish all relations and feasible ranges for each digit variable (0-9), ensuring consistency and no contradictions." 
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking5a, answer5a, thinking5b, answer5b], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, analyze combined digit-sum constraints and feasible ranges, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {
        "thinking": thinking6a,
        "answer": answer6a
    }
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    
    cot_reflect_instruction_6b = "Sub-task 6b: Systematically enumerate all possible digit assignments (a,b,c,d,e,f in 0-9) that satisfy the derived constraints from subtask 6a, without prematurely fixing any variables." 
    cot_agent_6b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6b = self.max_round
    cot_inputs_6b = [taskInfo, thinking6a, answer6a]
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_reflect_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "Reflexion"
    }
    thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_reflect_instruction_6b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, enumerate all valid digit assignments, thinking: {thinking6b.content}; answer: {answer6b.content}")
    for i in range(N_max_6b):
        feedback, correct = await critic_agent_6b([taskInfo, thinking6b, answer6b], "please review the enumeration of valid digit assignments and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6b.extend([thinking6b, answer6b, feedback])
        thinking6b, answer6b = await cot_agent_6b(cot_inputs_6b, cot_reflect_instruction_6b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6b.id}, refining enumeration, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {
        "thinking": thinking6b,
        "answer": answer6b
    }
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    
    cot_instruction_6c = "Sub-task 6c: Verify the completeness and correctness of the enumeration by cross-checking that all enumerated solutions satisfy the original horizontal and vertical sum equations and digit constraints." 
    cot_agent_6c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": cot_instruction_6c,
        "context": ["user query", "thinking of subtask 6b", "answer of subtask 6b", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6c, answer6c = await cot_agent_6c([taskInfo, thinking6b, answer6b, thinking3, answer3, thinking4, answer4], cot_instruction_6c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6c.id}, verify completeness and correctness of enumeration, thinking: {thinking6c.content}; answer: {answer6c.content}")
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {
        "thinking": thinking6c,
        "answer": answer6c
    }
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Count the total number of valid digit assignments found in subtask 6c that satisfy both the horizontal and vertical sum conditions and digit constraints." 
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 6c", "answer of subtask 6c"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6c, answer6c], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, count valid digit assignments, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
