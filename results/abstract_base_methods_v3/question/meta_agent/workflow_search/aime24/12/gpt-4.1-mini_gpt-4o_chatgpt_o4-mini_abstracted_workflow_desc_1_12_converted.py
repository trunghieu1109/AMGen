async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Express the complex number z in polar form using the given magnitude |z|=4, i.e., represent z as 4(cos θ + i sin θ) or 4e^(iθ), introducing the variable θ as the parameter to optimize over."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, express z in polar form, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2a = "Sub-task 2a: Expand the term (75 + 117i)z by substituting z = 4(cos θ + i sin θ), and express the result explicitly in terms of cos θ and sin θ, carefully separating real and imaginary parts."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, expand (75+117i)z, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = "Sub-task 2b: Expand the term (96 + 144i)/z by substituting z = 4(cos θ + i sin θ), rewrite 1/z as (1/4)(cos θ - i sin θ), and express the result explicitly in terms of cos θ and sin θ, carefully separating real and imaginary parts."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking1, answer1], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, expand (96+144i)/z, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_instruction_2c = "Sub-task 2c: Combine the results of subtasks 2a and 2b by adding their real parts and their imaginary parts separately, to obtain the full expression (75 + 117i)z + (96 + 144i)/z in terms of cos θ and sin θ."
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking2a, answer2a, thinking2b, answer2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, combine expansions from 2a and 2b, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_instruction_2d = "Sub-task 2d: Verify the combined expression from subtask 2c by substituting specific test values of θ (e.g., 0 and π/2) and comparing the symbolic expression's numeric evaluation with direct numeric computation of the original expression to ensure correctness."
    cot_agent_2d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": cot_instruction_2d,
        "context": ["user query", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "CoT"
    }
    thinking2d, answer2d = await cot_agent_2d([taskInfo, thinking2c, answer2c], cot_instruction_2d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2d.id}, verify combined expression by numeric substitution, thinking: {thinking2d.content}; answer: {answer2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Extract the real part of the combined expression obtained in subtask 2c, expressing it explicitly as a function of θ using trigonometric terms."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2d, answer2d], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, extract real part as function of θ, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4a = "Sub-task 4a: Rewrite the real part expression from subtask 3 in the form A cos θ + B sin θ by identifying coefficients A and B precisely."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, identify coefficients A and B, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {
        "thinking": thinking4a,
        "answer": answer4a
    }
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_instruction_4b = "Sub-task 4b: Calculate the amplitude R = sqrt(A^2 + B^2) and phase φ = arctan(B/A) of the trigonometric expression from subtask 4a, verifying the calculations by computing A^2 and B^2 separately and confirming the result."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4b.id}, calculate amplitude and phase with verification, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {
        "thinking": thinking4b,
        "answer": answer4b
    }
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Determine the value of θ that maximizes the real part expression by using the amplitude and phase from subtask 4b, recognizing that the maximum of A cos θ + B sin θ is R, and identifying the corresponding θ = -φ."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4b, answer4b], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4b, answer4b] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determine θ maximizing real part, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the value of θ that maximizes the real part.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining optimal θ, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Calculate the maximum value of the real part by substituting the optimal θ from subtask 5 back into the simplified expression or directly using the amplitude R, confirming the numeric result."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4b, answer4b, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculate maximum real part value, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
