async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Extract and clearly summarize all given numerical data and conditions from the problem statement, ensuring precise understanding of the sets involved (D, G, S, C) and the total population, without attempting any calculations or assumptions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting and summarizing data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)

    reflect_instruction_2 = "Sub-task 2: Clarify the interpretation of 'exactly two' and 'exactly three' items owned, explicitly confirming that candy hearts (owned by all) are included in these counts, and establish the implications of this for the problem's set structure. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": reflect_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, clarifying interpretation, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining interpretation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)

    cot_instruction_3 = "Sub-task 3: Compute the number of residents owning exactly one of the three items D, G, S (plus candy hearts), and the number owning none of D, G, S (only candy hearts), using the total population and given counts of exactly two and exactly three items owners, without yet solving for the number owning all four."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing exact one and none counts, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)

    cot_sc_instruction_4 = "Sub-task 4: Formally represent the sets D, G, S, and C, and define variables for the sizes of all possible intersections (including the number owning all four items), establishing equations based on inclusion-exclusion principles and the problem's given data."
    N_sc = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking and answers of subtasks 1,2,3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i](cot_inputs_4, cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, formalizing sets and variables, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent formal representation and equations for the problem.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing formal representation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)

    cot_sc_instruction_5 = "Sub-task 5: Derive and simplify the system of equations relating the sizes of single, double, triple, and quadruple intersections of D, G, and S (with C universal), using the counts of exactly one, two, three, and total owners, to isolate the variable representing the number owning all four items."
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking and answers of subtasks 1-4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i](cot_inputs_5, cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, deriving and simplifying equations, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_answers_5 + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent simplified system isolating the quadruple intersection.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing simplified system, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)

    reflect_instruction_6 = "Sub-task 6: Calculate the exact number of residents owning all four items (D, G, S, and C) by solving the derived equations, ensuring consistency with all given data and constraints, and verify the solution's correctness through logical checks. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflect_instruction_6,
        "context": ["user query", "thinking and answers of subtasks 1-5"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, calculating final number and verifying, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final calculation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
