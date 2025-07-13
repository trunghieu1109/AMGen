async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize problem constraints and analyze implications (CoT | SC_CoT)
    cot_instruction_0 = (
        "Sub-task 1: Formalize the problem constraints and analyze their implications on the list structure. "
        "Determine possible list lengths emphasizing that the median is a positive integer not in the list, implying even length. "
        "Represent constraints explicitly: sum=30; unique mode=9 with frequency strictly greater than others; median is integer absent from list; all elements positive integers with repetitions allowed except median. "
        "Avoid premature assumptions about fixed list length or element distinctness."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formalizing constraints, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    # Parse formalized constraints from answer_0 for use in enumeration
    # (In practice, parse or extract key points; here we pass as context)

    # Stage 1: Enumerate candidate lists and validate them

    # Subtask 1: Enumerate candidate lists (CoT | SC_CoT)
    cot_sc_instruction_1 = (
        "Sub-task 1: Enumerate candidate lists of positive integers summing to 30, with 9 as unique mode, "
        "for increasing even list lengths starting from 4 upwards. For each length, generate all possible pairs of middle elements whose average is a positive integer not in the list (median condition). "
        "For each median candidate, solve for remaining elements via integer partitions, allowing repeated elements except median, ensuring 9 appears strictly more times than any other number. "
        "Prepare comprehensive candidate lists for validation."
    )
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, enumerating candidates, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo] + possible_answers_1 + possible_thinkings_1,
        "Sub-task 1: Synthesize and choose the most comprehensive enumeration of candidate lists.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    # Subtask 2: Strict validation of candidate lists (CoT | SC_CoT | Debate)
    debate_instruction_2 = (
        "Sub-task 2: Strictly validate candidate lists from enumeration. Verify sum=30, unique mode=9 with frequency strictly greater than others, median is positive integer not in list, all elements positive integers, list length even. "
        "Reject candidates with tied modes or median elements in list. Cross-check candidates via debate among multiple agents to ensure robustness."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_1, answer_1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_1, answer_1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating candidates, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2(
        [taskInfo] + all_thinking_2[-1] + all_answer_2[-1],
        "Sub-task 2: Synthesize and finalize validated candidate lists after debate.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)

    # Subtask 3: Iterative feedback loop if no valid candidates (CoT | Reflexion)
    cot_reflect_instruction_3 = (
        "Sub-task 3: If no valid candidates remain after validation, iteratively revisit enumeration with increased list length or adjusted median candidates. "
        "Continue until at least one valid candidate is found or upper bound on list length is reached. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_1, answer_1, thinking_2, answer_2]
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, iterative refinement, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max_3):
        feedback_3, correct_3 = await critic_agent_3([taskInfo, thinking_3, answer_3],
                                                   "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback_3.content}; correct: {correct_3.content}")
        if correct_3.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback_3])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)

    # Subtask 4: Final selection of validated candidate list(s) (CoT | SC_CoT | Debate)
    debate_instruction_4 = (
        "Sub-task 4: From validated candidate lists, select final list(s) fully satisfying all constraints without ambiguity. "
        "Confirm uniqueness of mode 9, median condition, sum=30, positive integers. "
        "Cross-check and debate to ensure robustness. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking_2.content, answer_2.content, thinking_3.content, answer_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_2, answer_2, thinking_3, answer_3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_2, answer_2, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final selection, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4(
        [taskInfo] + all_thinking_4[-1] + all_answer_4[-1],
        "Sub-task 4: Synthesize and finalize the selected candidate list(s).",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Subtask 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)

    # Stage 2: Compute sum of squares and verify

    # Subtask 1: Compute sum of squares (CoT | SC_CoT)
    cot_instruction_5 = (
        "Sub-task 1: Compute the sum of the squares of all items in the finalized list from Stage 1. "
        "Carefully square each element and sum accurately."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_5 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_5, answer_5 = await cot_agent_5([taskInfo, thinking_4, answer_4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, computing sum of squares, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)

    # Subtask 2: Verify computed sum of squares (Reflexion | Debate)
    reflexion_instruction_6 = (
        "Sub-task 2: Verify the computed sum of squares by rechecking list elements and calculations. "
        "Confirm consistency with problem constraints and ensure no errors. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking_4, answer_4, thinking_5, answer_5]
    subtask_desc_6 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_6,
        "context": ["user query", thinking_4.content, answer_4.content, thinking_5.content, answer_5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, verifying sum of squares, thinking: {thinking_6.content}; answer: {answer_6.content}")
    for i in range(N_max_6):
        feedback_6, correct_6 = await critic_agent_6([taskInfo, thinking_6, answer_6],
                                                   "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback_6.content}; correct: {correct_6.content}")
        if correct_6.content == "True":
            break
        cot_inputs_6.extend([thinking_6, answer_6, feedback_6])
        thinking_6, answer_6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining verification, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)

    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
