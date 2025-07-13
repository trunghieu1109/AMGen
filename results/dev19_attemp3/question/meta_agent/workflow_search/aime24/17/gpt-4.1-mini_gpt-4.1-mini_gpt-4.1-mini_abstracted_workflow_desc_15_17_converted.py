async def forward_17(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 0_1: Rewrite and simplify the polynomial expression a^2b + a^2c + b^2a + b^2c + c^2a + c^2b "
        "using symmetric polynomial identities. Express it in terms of elementary symmetric sums or other simpler symmetric expressions to reduce complexity and facilitate further analysis."
    )
    cot_sc_agents_0_1 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, simplifying polynomial, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_0_1.append(thinking)
        possible_answers_0_1.append(answer)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 0_1: Synthesize and choose the most consistent and correct simplification of the polynomial expression.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 0_2: Based on the simplified polynomial expression from Sub-task 0_1, derive relationships between it and the linear sum constraint a + b + c = 300. "
        "Identify key parameters or invariants (such as symmetric sums S1, S2, S3) that characterize the solution set and rewrite the polynomial condition accordingly."
    )
    cot_sc_agents_0_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, deriving relations with sum constraint, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_0_2.append(thinking)
        possible_answers_0_2.append(answer)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2,
        "Sub-task 0_2: Synthesize and choose the most consistent and correct relationships between polynomial and sum constraint.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = (
        "Sub-task 1_1: Enumerate all triples (a,b,c) of nonnegative integers summing to 300 that satisfy the simplified polynomial condition derived in Stage 0. "
        "Explicitly partition the problem into exhaustive cases: (i) at least one variable zero, (ii) two variables equal, and (iii) all three variables distinct and positive. "
        "For each case, perform systematic enumeration or parameterization, and apply bounding or contradiction arguments to exclude impossible or extraneous solutions. "
        "This subtask must not assume uniqueness without rigorous proof or exhaustive search. " + reflect_inst_1_1
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_2, answer_0_2]
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_2, answer_0_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, enumerating valid triples, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(N_max_1_1):
        feedback, correct = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1],
                                                 "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, answer_1_1, feedback])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining enumeration, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = (
        "Sub-task 1_2: Verify the validity and uniqueness of each candidate triple (a,b,c) found in Subtask 1_1. Confirm that each satisfies both the sum and polynomial constraints exactly, and rigorously prove that no other solutions exist beyond those enumerated. "
        "This includes cross-validation of results, checking for overlooked cases, and ensuring no extraneous solutions are included. The verification should be iterative and allow revisiting enumeration if gaps are found." + reflect_inst_1_2
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking_1_1, answer_1_1]
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, verifying candidate triples, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    for i in range(N_max_1_2):
        feedback, correct = await critic_agent_1_2([taskInfo, thinking_1_2, answer_1_2],
                                                 "Please review and provide the limitations of provided verification. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2, answer_1_2, feedback])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining verification, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 2_1: Aggregate the count of all valid triples (a,b,c) confirmed in Stage 1 to produce the final number of solutions satisfying both constraints. "
        "Ensure that the aggregation only includes verified solutions and that the final count reflects the completeness and correctness established in previous subtasks."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_2, answer_1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, aggregating final count, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
