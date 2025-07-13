async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: select_and_verify_elements_under_constraints
    # Sub-task 1: Analyze median condition to determine possible list lengths and median values (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the implications of the median condition given the list sum is 30, mode is 9, "
        "and median is a positive integer not in the list. Deduce that the list length must be even, "
        "and enumerate possible pairs of middle elements whose average is an integer not in the list. "
        "Provide possible median values and list lengths consistent with these constraints."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0_1}")
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing median condition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Incorporate unique mode condition and frequency constraints (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Using the median analysis from Sub-task 1, incorporate the unique mode condition that 9 is the most frequent element. "
        "Determine the minimum frequency of 9 required to be the unique mode, and consider constraints on other elements' frequencies. "
        "Consider positivity and sum constraints to limit possible values and counts."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0_2}")
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing mode frequency, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Combine median and mode insights to propose candidate list lengths and partial distributions (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Combine insights from Sub-tasks 1 and 2 to propose candidate list lengths and partial element distributions. "
        "Use the sum constraint of 30 to restrict possible values and frequencies. Avoid assumptions contradicting positivity or unique mode. "
        "Prepare a framework for enumerating candidate lists in the next stage."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_0_3}")
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, combining median and mode insights, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: aggregate_valid_combinations
    # Sub-task 1: Enumerate all possible lists satisfying sum=30, unique mode=9, median integer not in list (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible lists of positive integers with sum 30, unique mode 9 with required frequency, "
        "and median as an integer not in the list (average of two middle elements). Verify median and mode conditions explicitly. "
        "Use systematic combinatorial exploration and pruning based on Stage 0 outputs."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Logging before agent calls: {subtask_desc_1_1}")
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_thinking_1_1, final_answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct candidate lists satisfying all conditions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {final_thinking_1_1.content}; answer - {final_answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": final_thinking_1_1, "answer": final_answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 2: Validate candidate lists from subtask 1 (SC_CoT + Reflexion)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Validate each candidate list from Sub-task 1 by checking sum, unique mode, positivity, and median conditions. "
        "Discard invalid candidates and confirm robustness and correctness of valid lists."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", final_thinking_1_1.content, final_answer_1_1.content],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    print(f"Logging before agent calls: {subtask_desc_1_2}")
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_2[i]([taskInfo, final_thinking_1_1, final_answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, validating candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_thinking_1_2, final_answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 2: Synthesize and confirm valid candidate lists after validation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {final_thinking_1_2.content}; answer - {final_answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": final_thinking_1_2, "answer": final_answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: aggregate_and_combine_values
    # Sub-task 1: Compute sum of squares of all items in each valid list (CoT)
    cot_instruction_2_1 = (
        "Sub-task 1: From the validated candidate lists in Stage 1, compute the sum of the squares of all items in each list. "
        "Identify the final sum of squares corresponding to the valid list(s)."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", final_thinking_1_2.content, final_answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    print(f"Logging before agent call: {subtask_desc_2_1}")
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, final_thinking_1_2, final_answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing sum of squares, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 2: Final verification and synthesis (Reflexion + CoT)
    reflect_inst_2_2 = (
        "Sub-task 2: Perform a final verification and synthesis step. Confirm that the computed sum of squares corresponds to a list that meets all problem criteria. "
        "Provide the final answer along with a concise explanation of why this solution is unique and valid."
    )
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Your problem is to find the sum of squares of a list of positive integers with sum 30, unique mode 9, "
        "and median integer not in the list. " + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    print(f"Logging before agent call: {subtask_desc_2_2}")
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying final answer, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2],
                                                  "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining final answer, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
