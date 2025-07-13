async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Derive and validate a simplified algebraic representation of the polynomial inside the product, i.e., express 2 - 2x + x^2 in a form that facilitates evaluation at roots of unity, with context from the query."
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, derive polynomial simplification, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    counter_1 = Counter([a.content.strip() for a in possible_answers_1])
    most_common_answer_1 = counter_1.most_common(1)[0][0]
    idx_1 = [a.content.strip() for a in possible_answers_1].index(most_common_answer_1)
    thinking1_final = possible_thinkings_1[idx_1]
    answer1_final = possible_answers_1[idx_1]
    subtask_desc1['response'] = {
        "thinking": thinking1_final,
        "answer": answer1_final
    }
    logs.append(subtask_desc1)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Analyze properties of the 13th roots of unity, including the minimal polynomial and cyclotomic polynomial, to understand how the polynomial evaluations relate to these roots, with context from the query."
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyze 13th roots properties, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    counter_2 = Counter([a.content.strip() for a in possible_answers_2])
    most_common_answer_2 = counter_2.most_common(1)[0][0]
    idx_2 = [a.content.strip() for a in possible_answers_2].index(most_common_answer_2)
    thinking2_final = possible_thinkings_2[idx_2]
    answer2_final = possible_answers_2[idx_2]
    subtask_desc2['response'] = {
        "thinking": thinking2_final,
        "answer": answer2_final
    }
    logs.append(subtask_desc2)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = "Sub-task 3: Combine the polynomial representation and roots of unity properties to express the product over k=0 to 12 of (2 - 2ω^k + ω^{2k}) as a single polynomial evaluation or a norm, simplifying the product into a manageable algebraic form. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final]
    subtask_desc3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, combine polynomial and roots properties, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        critic_inst_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], critic_inst_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content.strip() == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining combined expression, thinking: {thinking3.content}; answer: {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 4: Infer and compute the exact value of the simplified product expression, using algebraic identities, polynomial factorization, or norm computations, to obtain an explicit integer value, with context from the combined simplification in Sub-task 3."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, compute exact product value, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    counter_4 = Counter([a.content.strip() for a in possible_answers_4])
    most_common_answer_4 = counter_4.most_common(1)[0][0]
    idx_4 = [a.content.strip() for a in possible_answers_4].index(most_common_answer_4)
    thinking4_final = possible_thinkings_4[idx_4]
    answer4_final = possible_answers_4[idx_4]
    subtask_desc4['response'] = {
        "thinking": thinking4_final,
        "answer": answer4_final
    }
    logs.append(subtask_desc4)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Select and verify the final result by computing the remainder of the product value modulo 1000, ensuring correctness and consistency with the problem constraints, with context from the exact product value in Sub-task 4."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4_final, answer4_final],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4_final, answer4_final], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, compute remainder modulo 1000, thinking: {thinking5.content}; answer: {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
