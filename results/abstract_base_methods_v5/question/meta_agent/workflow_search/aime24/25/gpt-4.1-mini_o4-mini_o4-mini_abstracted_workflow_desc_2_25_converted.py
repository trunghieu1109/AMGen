async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the geometric properties of the convex equilateral hexagon ABCDEF, focusing on the implications of all pairs of opposite sides being parallel for the directions and angles of the hexagon's sides."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing geometric properties, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Express the side vectors of hexagon ABCDEF in terms of a single unknown side length s and direction angles, using the equilateral condition and parallelism of opposite sides to relate these vectors and establish angle notations (e.g., α, β, γ), based on the analysis from Sub-task 1."
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
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Determine the directions of the lines containing sides AB, CD, and EF, and find the angles between these lines based on the hexagon's properties and the vector expressions from Sub-task 2."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, determining directions and angles, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = "Sub-task 4: Relate the given side lengths (200, 240, 300) of the triangle formed by the extensions of sides AB, CD, and EF to the hexagon's side length s and the angles between these lines, preparing for trigonometric analysis, based on Sub-task 3 outputs."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, relating triangle side lengths, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Derive explicit formulas for the distances between pairs of parallel lines corresponding to sides AB, CD, and EF in terms of the hexagon side length s and the sine of the external angles α, β, and γ, avoiding unsupported assumptions about angle sums, based on Sub-task 4 outputs."
    cot_agent_5a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking5a, answer5a = await cot_agent_5a([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5a.id}, deriving distance formulas, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {
        "thinking": thinking5a,
        "answer": answer5a
    }
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Express each side length of the triangle formed by the extended lines (200, 240, 300) as s multiplied by the sine of the corresponding angle (e.g., 200 = s·sinβ, 240 = s·sinγ, 300 = s·sinα), based on the formulas derived in Sub-task 5a."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, expressing triangle sides as s times sine of angles, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {
        "thinking": thinking5b,
        "answer": answer5b
    }
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_5c = "Sub-task 5c: Summarize all known constraints and relations from previous subtasks, including angle sum relations (e.g., α + β + γ = 360°) and the equilateral condition, to form a consistent system of equations, based on Sub-task 5b outputs."
    cot_agent_5c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "CoT"
    }
    thinking5c, answer5c = await cot_agent_5c([taskInfo, thinking5b, answer5b], cot_instruction_5c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5c.id}, summarizing constraints and forming system of equations, thinking: {thinking5c.content}; answer: {answer5c.content}")
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {
        "thinking": thinking5c,
        "answer": answer5c
    }
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_instruction_5d = "Sub-task 5d: Solve the resulting system of trigonometric equations numerically to find the hexagon's side length s, ensuring no unsupported assumptions are made and verifying consistency by plugging solutions back into all equations, based on Sub-task 5c outputs."
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_instruction_5d,
        "context": ["user query", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "CoT"
    }
    thinking5d, answer5d = await cot_agent_5d([taskInfo, thinking5c, answer5c], cot_instruction_5d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5d.id}, solving system for side length s, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {
        "thinking": thinking5d,
        "answer": answer5d
    }
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    cot_reflect_instruction_5e = "Sub-task 5e: Perform a reflexive verification of the solution by independently checking assumptions, intermediate results, and the final value of s for logical consistency and agreement with all problem conditions, based on Sub-task 5d outputs."
    cot_agent_5e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5e = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5e = self.max_round
    cot_inputs_5e = [taskInfo, thinking5d, answer5d]
    subtask_desc5e = {
        "subtask_id": "subtask_5e",
        "instruction": cot_reflect_instruction_5e,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "Reflexion"
    }
    thinking5e, answer5e = await cot_agent_5e(cot_inputs_5e, cot_reflect_instruction_5e, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5e.id}, verifying solution consistency, thinking: {thinking5e.content}; answer: {answer5e.content}")
    for i in range(N_max_5e):
        feedback, correct = await critic_agent_5e([taskInfo, thinking5e, answer5e], "Please review the solution verification and provide any limitations or errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5e.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5e.extend([thinking5e, answer5e, feedback])
        thinking5e, answer5e = await cot_agent_5e(cot_inputs_5e, cot_reflect_instruction_5e, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5e.id}, refining verification, thinking: {thinking5e.content}; answer: {answer5e.content}")
    sub_tasks.append(f"Sub-task 5e output: thinking - {thinking5e.content}; answer - {answer5e.content}")
    subtask_desc5e['response'] = {
        "thinking": thinking5e,
        "answer": answer5e
    }
    logs.append(subtask_desc5e)
    print("Step 5e: ", sub_tasks[-1])
    
    debate_instruction_5f = "Sub-task 5f: Conduct a debate or consistency check to resolve any remaining conflicts or ambiguities in the solution, finalize the numeric value of the hexagon's side length s, and confirm it matches the problem's conditions and known results, based on Sub-task 5e outputs."
    debate_agents_5f = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5f = self.max_round
    all_thinking5f = [[] for _ in range(N_max_5f)]
    all_answer5f = [[] for _ in range(N_max_5f)]
    subtask_desc5f = {
        "subtask_id": "subtask_5f",
        "instruction": debate_instruction_5f,
        "context": ["user query", "thinking of subtask 5e", "answer of subtask 5e"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5f):
        for i, agent in enumerate(debate_agents_5f):
            if r == 0:
                thinking5f, answer5f = await agent([taskInfo, thinking5e, answer5e], debate_instruction_5f, r, is_sub_task=True)
            else:
                input_infos_5f = [taskInfo, thinking5e, answer5e] + all_thinking5f[r-1] + all_answer5f[r-1]
                thinking5f, answer5f = await agent(input_infos_5f, debate_instruction_5f, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, finalizing side length s, thinking: {thinking5f.content}; answer: {answer5f.content}")
            all_thinking5f[r].append(thinking5f)
            all_answer5f[r].append(answer5f)
    final_decision_agent_5f = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5f, answer5f = await final_decision_agent_5f([taskInfo] + all_thinking5f[-1] + all_answer5f[-1], "Sub-task 5f: Make final decision on the numeric value of the hexagon's side length s.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final side length s, thinking: {thinking5f.content}; answer: {answer5f.content}")
    sub_tasks.append(f"Sub-task 5f output: thinking - {thinking5f.content}; answer - {answer5f.content}")
    subtask_desc5f['response'] = {
        "thinking": thinking5f,
        "answer": answer5f
    }
    logs.append(subtask_desc5f)
    print("Step 5f: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5f, answer5f, sub_tasks, agents)
    return final_answer, logs
