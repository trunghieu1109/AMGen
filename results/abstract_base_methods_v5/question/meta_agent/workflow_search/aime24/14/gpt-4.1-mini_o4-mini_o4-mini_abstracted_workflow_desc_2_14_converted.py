async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Subtask 1: Parameterize points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1 using hyperbolic functions, ensuring each point satisfies the hyperbola equation and clearly defining the parameter domain." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, parameterize points A, B, C, D on hyperbola, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Subtask 2: Formulate the geometric conditions for ABCD to be a rhombus with diagonals intersecting at the origin, including equal side lengths and midpoint of diagonals at the origin, expressed in terms of the parameters from Subtask 1." 
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulate rhombus conditions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Subtask 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Subtask 3: Derive algebraic relations between the parameters of points A, B, C, and D based on the rhombus conditions and hyperbola parameterization, focusing on expressing the diagonals and side lengths explicitly." 
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, derive algebraic relations, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the algebraic relations derivation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining algebraic relations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = "Subtask 4: Express the squared length of diagonal BD (BD^2) as a function of the parameters, using the relations obtained in Subtask 3." 
    N4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, express BD^2 in parameters, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Subtask 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Subtask 5a: Analyze the domain and constraints of the parameters involved in the expression for BD^2, explicitly stating parameter ranges and their geometric implications for the rhombus on the hyperbola." 
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, analyze parameter domain and constraints, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Subtask 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Subtask 5b: Determine the monotonicity of the function BD^2 with respect to the relevant parameter(s) by computing and analyzing the sign of its derivative(s) over the domain established in Subtask 5a." 
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking4, answer4, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, analyze monotonicity of BD^2, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Subtask 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_5c = "Subtask 5c: Evaluate the limits of BD^2 at the boundaries of the parameter domain identified in Subtask 5a to identify candidate extremal values." 
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5c, answer5c = await cot_agent_5c([taskInfo, thinking4, answer4, thinking5a, answer5a], cot_instruction_5c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5c.id}, evaluate limits of BD^2 at domain boundaries, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Subtask 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    reflexion_instruction_5d = "Subtask 5d: Reconcile and verify the results from Subtasks 5b and 5c through a Reflexion step that cross-checks monotonicity, limits, and assumptions to resolve any conflicting conclusions and determine the minimal upper bound of BD^2." 
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5d = self.max_round
    cot_inputs_5d = [taskInfo, thinking5b, answer5b, thinking5c, answer5c]
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": reflexion_instruction_5d,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Reflexion"
    }
    thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, reflexion_instruction_5d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, reconcile monotonicity and limits, thinking: {thinking5d.content}; answer: {answer5d.content}")
    for i in range(N_max_5d):
        feedback, correct = await critic_agent_5d([taskInfo, thinking5d, answer5d], "please review the reconciliation of monotonicity and limits and verify minimal upper bound determination.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5d.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5d.extend([thinking5d, answer5d, feedback])
        thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, reflexion_instruction_5d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, refining reconciliation, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Subtask 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    cot_instruction_6 = "Subtask 6: Verify the feasibility and consistency of the minimal upper bound of BD^2 found in Subtask 5d with the hyperbola and rhombus conditions, ensuring it applies to all such rhombi." 
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5d, answer5d], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, verify bound feasibility, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_sc_instruction_7 = "Subtask 7: Interpret the query’s requirement precisely and return the greatest real number less than BD^2 for all such rhombi, using a Self-Consistency Chain-of-Thought (SC-CoT) approach to consider possible interpretations and select the answer that best matches the original query." 
    N7 = self.max_sc
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible_answers_7 = []
    thinkingmapping_7 = {}
    answermapping_7 = {}
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N7):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6, answer6], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, interpret and compute greatest real number less than BD^2, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7.content)
        thinkingmapping_7[answer7.content] = thinking7
        answermapping_7[answer7.content] = answer7
    answer7_content = Counter(possible_answers_7).most_common(1)[0][0]
    thinking7 = thinkingmapping_7[answer7_content]
    answer7 = answermapping_7[answer7_content]
    sub_tasks.append(f"Subtask 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Subtask 8: Perform a final verification of the numeric answer from Subtask 7 against the original problem statement and geometric feasibility, confirming correctness before output." 
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, final verification of numeric answer, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {
        "thinking": thinking8,
        "answer": answer8
    }
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
