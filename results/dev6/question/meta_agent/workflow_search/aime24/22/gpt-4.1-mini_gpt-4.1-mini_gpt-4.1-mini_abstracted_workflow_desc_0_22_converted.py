async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze and formalize the given constraints on the list: sum = 30, unique mode = 9, "
        "median is a positive integer not in the list. Determine implications for list length (must be even), "
        "median properties (median is average of two consecutive integers not in the list), and mode frequency (frequency of 9 greater than any other element). "
        "Identify necessary assumptions such as the list being sorted for median calculation and that all elements are positive integers. Avoid assuming list length without justification. "
        "This subtask sets the foundation for selecting candidate elements and their frequencies."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_1: {subtask_desc1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Generate candidate lists of positive integers that satisfy sum=30, have even length (from subtask_1), "
        "contain 9 with frequency greater than any other number, and have two middle elements whose average is the median (a positive integer not in the list). "
        "Enumerate possible values for the two middle elements (consecutive integers) and other elements ensuring sum and mode constraints hold. "
        "Avoid including the median value itself in the list. This involves combinatorial reasoning and verification of constraints."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before subtask_4: {subtask_desc4}")
    possible_answers = []
    possible_thinkings = []
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, generating candidate lists, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4)
        possible_thinkings.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Synthesize and choose the most consistent and correct candidate lists satisfying all constraints."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers + possible_thinkings, final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Verify each candidate list from subtask_4 for all conditions: sum equals 30, unique mode is 9, median is a positive integer not in the list, and all elements are positive integers. "
        "Discard invalid candidates. This verification ensures only valid lists proceed to the final calculation."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4, answer4],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_5: {subtask_desc5}")
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, verifying candidate lists, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: For the validated list(s) from subtask_5, compute the sum of the squares of all items. "
        "Square each element and sum these values. If multiple valid lists exist, verify if the sum of squares is unique or if further refinement is needed."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", thinking5, answer5],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before subtask_6: {subtask_desc6}")
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, computing sum of squares, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_7 = (
        "Sub-task 7: Perform a final verification of the computed sum of squares against all problem constraints to ensure consistency and correctness. "
        "If inconsistencies arise, revisit previous subtasks for refinement. Provide the final answer with justification."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6, thinking5, answer5]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6, answer6, thinking5, answer5],
        "agent_collaboration": "Reflexion"
    }
    print(f"Logging before subtask_7: {subtask_desc7}")
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, final verification, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7],
                                               "Please review and provide the limitations of provided solutions. If correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining solution, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
