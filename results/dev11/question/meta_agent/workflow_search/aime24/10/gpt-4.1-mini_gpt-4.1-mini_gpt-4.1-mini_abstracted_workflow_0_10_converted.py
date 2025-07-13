async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally represent the geometric configuration of rectangles ABCD and EFGH. "
        "Assign coordinate systems or vector notations consistent with rectangle properties and given side lengths: AB=107, BC=16 for ABCD; EF=184, FG=17 for EFGH. "
        "Explicitly incorporate the collinearity constraint of points D, E, C, and F by expressing their coordinates on a single line. "
        "Represent the cyclic condition on points A, D, H, and G using circle equations or relevant properties (e.g., equal angles, power of a point). "
        "Avoid assuming orientations or orders not implied by the problem; instead, state any reasonable assumptions explicitly and justify them. "
        "The goal is to establish a consistent algebraic framework capturing all given constraints without premature numeric substitution."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formal geometric representation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the formal coordinate representation from Sub-task 1, derive symbolic geometric relations. "
        "Use rectangle properties (right angles, equal opposite sides) to relate coordinates of points within each rectangle. "
        "Use the collinearity of points D, E, C, and F to establish linear equations linking points from both rectangles. "
        "Apply the cyclic condition on points A, D, H, and G to form circle equations or angle equalities. "
        "Summarize these relations as a consistent system of symbolic equations linking unknown coordinates and distances. "
        "Avoid premature numeric substitution and refrain from introducing assumptions not supported by the problem or Sub-task 1. "
        "Emphasize clarity in expressing these relations to facilitate numeric substitution in later stages."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, derive symbolic relations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Synthesize and choose the most consistent and correct symbolic relations for the geometric configuration."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, final_instr_2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_reflect_instruction_3 = (
        "Sub-task 3: Substitute the given numeric side lengths (BC=16, AB=107, FG=17, EF=184) into the symbolic relations derived previously. "
        "Use the linear (collinearity) and cyclic constraints to express the coordinates or distances involving points C and E in terms of known quantities. "
        "Solve the resulting system of equations to isolate an explicit expression for the segment CE. "
        "Carefully analyze all numeric roots obtained, discard extraneous or non-geometric solutions based on plausibility and geometric reasoning, and clearly justify the choice of the unique valid numeric solution. "
        "Emphasize the uniqueness and consistency of this solution with all prior constraints. "
        "Avoid introducing unjustified assumptions or reverting to earlier flawed assumptions about point positions. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1.content, answer1.content, thinking2.content, answer2.content]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, numeric substitution and solution, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3.content, answer3.content],
                                               "Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3.content, answer3.content, feedback.content])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining numeric solution, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    cot_reflect_instruction_4a = (
        "Sub-task 4a: Verify the numeric solution for CE obtained in Sub-task 3 against all geometric constraints, especially the cyclic condition on points A, D, H, G, and the collinearity of D, E, C, F. "
        "Explicitly use the numeric values and reasoning from Sub-task 3 as the starting point. Do not revert to earlier assumptions unless rigorously justified. "
        "Check for consistency and plausibility of the numeric solution, ensuring that the cyclic quadrilateral properties and rectangle side relations hold numerically. "
        "If inconsistencies are found, refine the numeric solution accordingly before proceeding. "
        "Provide a detailed verification report alongside the candidate numeric value for CE. "
        "Use answer3 as your candidate and do not re-derive from scratch."
    )
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4a = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4a = self.max_round
    cot_inputs_4a = [taskInfo, thinking3.content, answer3.content]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_reflect_instruction_4a,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, verifying numeric solution, thinking: {thinking4a.content}; answer: {answer4a.content}")
    for i in range(N_max_4a):
        feedback, correct = await critic_agent_4a([taskInfo, thinking4a.content, answer4a.content, thinking3.content, answer3.content],
                                                 "Please rigorously check consistency between this answer and Sub-task 3 numeric solution. Reject any contradictions. If consistent, output exactly 'True' in 'correct'.",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4a.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4a.extend([thinking4a.content, answer4a.content, feedback.content])
        thinking4a, answer4a = await cot_agent_4a(cot_inputs_4a, cot_reflect_instruction_4a, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4a.id}, refining verification, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)

    cot_reflect_instruction_4b = (
        "Sub-task 4b: Finalize the numeric computation of CE using the verified expressions and values from Sub-task 4a. "
        "Provide a concise, clearly justified final numeric answer for CE. "
        "Include a summary of the verification results confirming that the final answer satisfies all geometric and cyclic constraints. "
        "Ensure no re-derivation from scratch occurs here; instead, rely on the verified candidate from prior subtasks. "
        "Present final consistency checks or assertions comparing this answer to Sub-task 3 results to guarantee alignment. "
        "Return the final numeric value of CE along with verification confirmation."
    )
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4b = [taskInfo, thinking4a.content, answer4a.content, thinking3.content, answer3.content]
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", thinking4a.content, answer4a.content, thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, finalizing numeric answer, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)

    final_answer = await self.make_final_answer(thinking4b, answer4b, sub_tasks, agents)
    return final_answer, logs