async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Subtask 1: Analyze the geometric properties of the convex equilateral hexagon ABCDEF, specifically that all pairs of opposite sides are parallel, and deduce the directions of the sides and the implications for the hexagon's shape and side vectors." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Subtask 2: Express the side vectors of hexagon ABCDEF in terms of a common side length s and three unit direction vectors u, v, w, corresponding to sides AB, CD, and EF respectively, using the parallelism and equilateral conditions to relate opposite sides and establish vector relations." 
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, expressing side vectors, thinking: {thinking2.content}; answer: {answer2.content}")
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
    
    cot_reflect_instruction_3 = "Subtask 3: Fix a concrete coordinate system by setting point A at the origin and defining explicit unit vectors u, v, w for directions of sides AB, CD, and EF, then express points C and E in terms of s and these vectors to fully parametrize the hexagon vertices relevant to the problem. Verify and refine these parametrizations through reflexion." 
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
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, fixing coordinate system and parametrizing points, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the coordinate system and parametrizations for completeness and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining parametrizations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4a = "Subtask 4a: Derive explicit parametric equations for the lines extending sides AB, CD, and EF using the fixed coordinate system and parametrized points from Subtask 3, expressing each line in vector form suitable for intersection calculations." 
    N = self.max_sc
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, deriving parametric line equations, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Subtask 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_sc_instruction_4b = "Subtask 4b: Compute the intersection points P = AB ∩ CD, Q = CD ∩ EF, and R = EF ∩ AB explicitly in coordinates as functions of the side length s, deriving closed-form expressions for the triangle side lengths PQ(s), QR(s), and RP(s) in the form α₁·s, α₂·s, and α₃·s. Use multiple independent methods to confirm correctness." 
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_sc_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_sc_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, computing intersection points and side length coefficients, thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Subtask 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    cot_instruction_5a = "Subtask 5a: Formulate the system of equations by equating the computed triangle side lengths PQ(s), QR(s), and RP(s) to the given lengths 200, 240, and 300 respectively, resulting in explicit algebraic equations in terms of s and coefficients α₁, α₂, α₃." 
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4b, answer4b], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, formulating system of equations for side length s, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Subtask 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    debate_instruction_5b = "Subtask 5b: Solve the system of equations rigorously to find the unique numeric value of the hexagon side length s, using algebraic or numerical methods. Engage in a debate among agents to argue and verify the solution, followed by reflexion to reconcile any conflicts." 
    debate_agents_5b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5b = self.max_round
    all_thinking5b = [[] for _ in range(N_max_5b)]
    all_answer5b = [[] for _ in range(N_max_5b)]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": debate_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5b):
        for i, agent in enumerate(debate_agents_5b):
            if r == 0:
                thinking5b, answer5b = await agent([taskInfo, thinking5a, answer5a], debate_instruction_5b, r, is_sub_task=True)
            else:
                input_infos_5b = [taskInfo, thinking5a, answer5a] + all_thinking5b[r-1] + all_answer5b[r-1]
                thinking5b, answer5b = await agent(input_infos_5b, debate_instruction_5b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system and finding hexagon side length, thinking: {thinking5b.content}; answer: {answer5b.content}")
            all_thinking5b[r].append(thinking5b)
            all_answer5b[r].append(answer5b)
    final_decision_agent_5b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5b, answer5b = await final_decision_agent_5b([taskInfo] + all_thinking5b[-1] + all_answer5b[-1], "Subtask 5b: Make final decision on the numeric value of the hexagon's side length s.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final hexagon side length, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Subtask 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_6 = "Subtask 6: Verify the solution for s by substituting it back into the expressions for PQ(s), QR(s), and RP(s) to confirm that the triangle side lengths match the given values 200, 240, and 300, ensuring internal consistency and correctness of the solution." 
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4b, answer4b, thinking5b, answer5b], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, verifying solution consistency, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
