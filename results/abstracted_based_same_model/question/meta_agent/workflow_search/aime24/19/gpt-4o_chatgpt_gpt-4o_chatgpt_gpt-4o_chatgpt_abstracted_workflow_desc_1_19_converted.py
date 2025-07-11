async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify the properties of the 13th roots of unity, specifically focusing on the fact that they satisfy the equation x^13 = 1 and the sum of all 13th roots of unity is zero. Additionally, understand the identity ∏_{k=0}^{n−1}(a−ω^k)=a^n−1 for roots of unity."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying properties of 13th roots of unity, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    debate_instruction_2a = "Sub-task 2a: Show that the product P = ∏_{k=0}^{12} (2 - 2ω^k + ω^{2k}) can be interpreted as the resultant of the polynomial f(x) = x^2 - 2x + 2 and x^13 - 1."
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking2a = [[] for _ in range(N_max_2a)]
    all_answer2a = [[] for _ in range(N_max_2a)]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": debate_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                thinking2a, answer2a = await agent([taskInfo, thinking1, answer1], 
                                           debate_instruction_2a, r, is_sub_task=True)
            else:
                input_infos_2a = [taskInfo, thinking1, answer1] + all_thinking2a[r-1] + all_answer2a[r-1]
                thinking2a, answer2a = await agent(input_infos_2a, debate_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, interpreting product as resultant, thinking: {thinking2a.content}; answer: {answer2a.content}")
            all_thinking2a[r].append(thinking2a)
            all_answer2a[r].append(answer2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking2a, answer2a = await final_decision_agent_2a([taskInfo] + all_thinking2a[-1] + all_answer2a[-1], 
                                                 "Sub-task 2a: Make final decision on interpretation as resultant.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, interpreting product as resultant, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    cot_instruction_2b = "Sub-task 2b: Factor the polynomial f(x) = x^2 - 2x + 2 into linear factors (x - (1 - i))(x - (1 + i))."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, factoring polynomial, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    cot_instruction_2c = "Sub-task 2c: Apply the identity ∏_{k=0}^{12}(ω^k - a) = a^13 - 1 for a = 1 ± i, then multiply the results to find the product P."
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_instruction_2c,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    thinking2c, answer2c = await cot_agent_2c([taskInfo, thinking2b, answer2b], cot_instruction_2c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2c.id}, applying identity and multiplying results, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    cot_reflect_instruction_3 = "Sub-task 3: Evaluate the expression obtained from subtask 2c to find its numerical value."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2c, answer2c]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, evaluating expression, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], 
                                       "please review the evaluation and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining evaluation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    debate_instruction_4 = "Sub-task 4: Compute the remainder of the numerical value obtained in subtask 3 when divided by 1000."
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
            agents.append(f"Debate agent {agent.id}, round {r}, computing remainder, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 
                                                 "Sub-task 4: Make final decision on remainder.", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating remainder, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
