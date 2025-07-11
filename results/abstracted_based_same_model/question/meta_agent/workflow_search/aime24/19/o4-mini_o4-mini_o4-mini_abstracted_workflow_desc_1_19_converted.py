async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Define the quadratic polynomial f(x)=x^2 - 2x + 2 that appears inside the product, identifying f so that ∏_{k=0..12}(2 - 2ω^k + ω^{2k}) = ∏_{k}f(ω^k)."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1","instruction": cot_instruction,"context": ["user query"],"agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, define f(x)=x^2 - 2x + 2, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction = "Sub-task 2: Recognize that the product ∏_{k=0..12} f(ω^k) equals the resultant Res(x^13 - 1, f(x))."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2","instruction": cot_sc_instruction,"context": ["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, recognize resultant relationship, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_reflect_instruction = "Sub-task 3: Apply the resultant formula Res(g,f)=a_m^n ∏_{β: f(β)=0} g(β) for g(x)=x^13 - 1 and f(x), reducing the product to ∏_{β roots of f}(β^13 - 1)."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc3 = {"subtask_id": "subtask_3","instruction": cot_reflect_instruction,"context": ["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"],"agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, reduce product via resultant formula, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "please review the reduction step and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, feedback], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, revised reduction, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    cot_instruction4 = "Sub-task 4: Compute the two roots β1, β2 of f(x)=x^2 - 2x + 2 using the quadratic formula."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4","instruction": cot_instruction4,"context": ["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, compute roots, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    cot_sc_instruction5 = "Sub-task 5: Compute A = β1^13 - 1 for the first root β1 obtained."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {"subtask_id": "subtask_5","instruction": cot_sc_instruction5,"context": ["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration": "SC_CoT"}
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, compute A, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers5.append(answer5_i.content)
        thinkingmapping5[answer5_i.content] = thinking5_i
        answermapping5[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    cot_sc_instruction6 = "Sub-task 6: Compute B = β2^13 - 1 for the second root β2 obtained."
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6","instruction": cot_sc_instruction6,"context": ["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration": "SC_CoT"}
    for i in range(N6):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking4, answer4], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, compute B, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers6.append(answer6_i.content)
        thinkingmapping6[answer6_i.content] = thinking6_i
        answermapping6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    cot_instruction7 = "Sub-task 7: Multiply A and B to obtain the resultant P = A * B."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7","instruction": cot_instruction7,"context": ["user query","thinking of subtask 5","answer of subtask 5","thinking of subtask 6","answer of subtask 6"],"agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, multiply A and B, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    cot_instruction8 = "Sub-task 8: Reduce the integer P modulo 1000 to find the requested remainder."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8","instruction": cot_instruction8,"context": ["user query","thinking of subtask 7","answer of subtask 7"],"agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, reduce modulo 1000, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)

    cot_instruction9 = "Sub-task 9: Return ONLY the integer remainder obtained from subtask 8."
    cot_agent9 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {"subtask_id": "subtask_9","instruction": cot_instruction9,"context": ["user query","thinking of subtask 8","answer of subtask 8"],"agent_collaboration": "CoT"}
    thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8], cot_instruction9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent9.id}, final formatting, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])
    subtask_desc9["response"] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs