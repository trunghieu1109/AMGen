async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Compute the area of triangular face ABC symbolically using Heron's formula with edges AB=sqrt(41), BC=sqrt(89), AC=sqrt(80). "
        "Show detailed Chain-of-Thought reasoning, compute semi-perimeter symbolically, calculate area squared, simplify radicals exactly (e.g., sqrt(756) as 6*sqrt(21)), and provide the exact simplified area expression without numerical approximations."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, computing area of face ABC, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Compute the area of triangular face ABD symbolically using Heron's formula with edges AB=sqrt(41), BD=sqrt(80), AD=sqrt(89). "
        "Provide detailed Chain-of-Thought reasoning, symbolic semi-perimeter, area squared, exact radical simplification, and final exact area expression without numerical approximations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, computing area of face ABD, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Compute the area of triangular face ACD symbolically using Heron's formula with edges AC=sqrt(80), CD=sqrt(41), AD=sqrt(89). "
        "Provide detailed Chain-of-Thought reasoning, symbolic semi-perimeter, area squared, exact radical simplification, and final exact area expression without numerical approximations."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing area of face ACD, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Compute the area of triangular face BCD symbolically using Heron's formula with edges BC=sqrt(89), CD=sqrt(41), BD=sqrt(80). "
        "Provide detailed Chain-of-Thought reasoning, symbolic semi-perimeter, area squared, exact radical simplification, and final exact area expression without numerical approximations."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, computing area of face BCD, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflexion_instruction_5 = (
        "Sub-task 5: Validate symbolically whether the four computed face areas from Sub-tasks 1 to 4 are equal or not. "
        "Compare their exact simplified radical forms without numerical approximations. Provide detailed reasoning and final conclusion on equality."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, validating face area equality, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the symbolic validation of face area equality and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining validation of face area equality, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Calculate the volume of tetrahedron ABCD using the Cayley-Menger determinant with the given edge lengths. "
        "Provide detailed Chain-of-Thought reasoning, symbolic determinant calculation, exact radical simplification, and final exact volume expression. "
        "Confirm consistency with the previously computed face areas."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing volume, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Calculate the total surface area of tetrahedron ABCD by summing the four exact symbolic face areas computed in Sub-tasks 1 to 4. "
        "Provide detailed Chain-of-Thought reasoning and final exact total surface area expression without assumptions of symmetry or equality."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", answer1.content, answer2.content, answer3.content, answer4.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, answer1, answer2, answer3, answer4], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, computing total surface area, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Compute the inradius (distance from point I to each face) of tetrahedron ABCD using the formula: inradius = 3 * volume / total surface area. "
        "Use the exact symbolic volume from Sub-task 6 and total surface area from Sub-task 7. Provide detailed Chain-of-Thought reasoning and final exact inradius expression, maintaining radical simplifications and avoiding numerical approximations."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", answer6.content, answer7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, answer6, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, computing inradius, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_sc_instruction_9 = (
        "Sub-task 9: Express the inradius computed in Sub-task 8 in the form (m * sqrt(n)) / p, where m and p are positive integers that are coprime, and n is a positive square-free integer. "
        "Simplify radicals and fractions exactly. Then compute m + n + p. Use Self-Consistency Chain-of-Thought to consider multiple simplifications and select the most consistent answer. "
        "Provide detailed reasoning and final simplified expression and sum."
    )
    N = self.max_sc
    cot_agents_9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_9 = []
    thinkingmapping_9 = {}
    answermapping_9 = {}
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_sc_instruction_9,
        "context": ["user query", thinking8.content, answer8.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking9, answer9 = await cot_agents_9[i]([taskInfo, thinking8, answer8], cot_sc_instruction_9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_9[i].id}, simplifying inradius expression, thinking: {thinking9.content}; answer: {answer9.content}")
        possible_answers_9.append(answer9.content)
        thinkingmapping_9[answer9.content] = thinking9
        answermapping_9[answer9.content] = answer9
    answer9_content = Counter(possible_answers_9).most_common(1)[0][0]
    thinking9 = thinkingmapping_9[answer9_content]
    answer9 = answermapping_9[answer9_content]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    reflexion_instruction_10 = (
        "Sub-task 10: Perform a consistency check comparing the computed inradius from Sub-task 8 and the face areas from Sub-tasks 1 to 4. "
        "Verify that the inradius and face areas satisfy known geometric relations for tetrahedra. Flag any discrepancies for review. Provide detailed reasoning and final conclusion."
    )
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_10 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_10 = self.max_round
    cot_inputs_10 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking8, answer8, thinking9, answer9]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": reflexion_instruction_10,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content, thinking8.content, answer8.content, thinking9.content, answer9.content],
        "agent_collaboration": "Reflexion"
    }
    thinking10, answer10 = await cot_agent_10(cot_inputs_10, reflexion_instruction_10, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_10.id}, performing consistency check, thinking: {thinking10.content}; answer: {answer10.content}")
    for i in range(N_max_10):
        feedback, correct = await critic_agent_10([taskInfo, thinking10, answer10], "please review the consistency check and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_10.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_10.extend([thinking10, answer10, feedback])
        thinking10, answer10 = await cot_agent_10(cot_inputs_10, reflexion_instruction_10, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_10.id}, refining consistency check, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    reflexion_instruction_11 = (
        "Sub-task 11: Verify the correctness of the computed inradius expression and the final sum m + n + p from Sub-task 9. "
        "Provide a final confirmation, summary of verification results from previous subtasks, and confidence in the solution. "
        "Provide the final answer as requested."
    )
    cot_agent_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_11 = self.max_round
    cot_inputs_11 = [taskInfo, thinking9, answer9, thinking10, answer10]
    subtask_desc11 = {
        "subtask_id": "subtask_11",
        "instruction": reflexion_instruction_11,
        "context": ["user query", thinking9.content, answer9.content, thinking10.content, answer10.content],
        "agent_collaboration": "Reflexion"
    }
    thinking11, answer11 = await cot_agent_11(cot_inputs_11, reflexion_instruction_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_11.id}, verifying final answer, thinking: {thinking11.content}; answer: {answer11.content}")
    for i in range(N_max_11):
        feedback, correct = await critic_agent_11([taskInfo, thinking11, answer11], "please review the final answer correctness and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_11.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_11.extend([thinking11, answer11, feedback])
        thinking11, answer11 = await cot_agent_11(cot_inputs_11, reflexion_instruction_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_11.id}, refining final answer, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {"thinking": thinking11, "answer": answer11}
    logs.append(subtask_desc11)
    print("Step 11: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs
