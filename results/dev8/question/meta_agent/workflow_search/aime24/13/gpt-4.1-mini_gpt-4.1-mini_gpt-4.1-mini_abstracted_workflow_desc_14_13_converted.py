async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 0.1: Clarify the exact geometric configuration of the chain of tangent circles inside triangle ABC at vertex B. "
        "Determine precisely which two circles are tangent to sides AB and BC, respectively, and whether these circles are the first and last in the chain or positioned differently. "
        "Establish whether the chain is bounded by these sides or simply lies between them. Avoid assuming the chain starts or ends at these sides without justification. "
        "Use geometric reasoning or auxiliary constructions to resolve ambiguities in the problem statement. This step is crucial to prevent misinterpretation of the chain's placement and to set a correct foundation for subsequent derivations."
    )
    N_sc_0 = self.max_sc
    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc0 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, clarifying geometric configuration, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_answers_0 + possible_thinkings_0, "Sub-task 0.1: Synthesize and choose the most consistent geometric configuration clarification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0.1: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 0.2: Formulate the problem setting symbolically by defining variables for the angle at vertex B, the radii of the circles in the chain, "
        "and their positions relative to vertex B and sides AB and BC. Establish notation for the number of circles (n), the radius of the first circle in the chain (R), "
        "and the progression of radii if any. Avoid introducing unverified formulas; instead, prepare for first-principles derivation. "
        "This subtask sets up the symbolic framework needed for rigorous geometric analysis."
    )
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc1 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking1, answer1 = await cot_sc_agents_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, formulating symbolic variables, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, "Sub-task 0.2: Synthesize and choose the most consistent symbolic formulation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1.1: Derive from first principles the geometric relationships governing the chain of tangent circles inscribed in the angle at vertex B. "
        "Use trigonometric laws (law of sines, law of cosines) and properties of tangent circles to express the distances from vertex B to each circle's center and the radii progression. "
        "Avoid applying shortcut formulas without derivation. Develop symbolic expressions relating the angle at B, the number of circles, their radii, and the inradius of triangle ABC. "
        "Maintain exact symbolic forms and postpone numeric evaluation until all relationships are established and validated."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1, answer1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, deriving geometric relationships, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 1.2: Using the derived symbolic relationships, compute the measure of the angle at vertex B from the first configuration (8 circles of radius 34). "
        "Then, independently verify that this angle is consistent with the second configuration (2024 circles of radius 1). "
        "This step ensures that the geometric model fits both given data sets and that the angle is physically plausible. "
        "If inconsistencies arise, revisit assumptions and derivations. This subtask is critical for validating the correctness of the geometric model before proceeding."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, computing and verifying angle at B, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc2_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2_3 = (
        "Sub-task 1.3: Validate the geometric consistency of the derived parameters (angle at B, radii progression, distances) across both circle configurations. "
        "Check that the chain of circles can be inscribed in the angle with the computed parameters and that the inradius implied by these parameters is consistent. "
        "If inconsistencies or contradictions are found, trigger a re-examination of the derivations or assumptions. "
        "This subtask acts as a quality control checkpoint to ensure the solution path remains valid."
    )
    cot_sc_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_2_3 = []
    possible_thinkings_2_3 = []
    subtask_desc2_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_sc_instruction_2_3,
        "context": ["user query", thinking2_1.content, answer2_1.content, thinking2_2.content, answer2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking2_3, answer2_3 = await cot_sc_agents_2_3[i]([taskInfo, thinking2_1, answer2_1, thinking2_2, answer2_2], cot_sc_instruction_2_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_3[i].id}, validating geometric consistency, thinking: {thinking2_3.content}; answer: {answer2_3.content}")
        possible_answers_2_3.append(answer2_3)
        possible_thinkings_2_3.append(thinking2_3)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_3, answer2_3 = await final_decision_agent_2_3([taskInfo] + possible_answers_2_3 + possible_thinkings_2_3, "Sub-task 1.3: Synthesize and confirm geometric consistency.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}")
    subtask_desc2_3['response'] = {"thinking": thinking2_3, "answer": answer2_3}
    logs.append(subtask_desc2_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 2.1: Compute the inradius of triangle ABC using the validated geometric relationships and parameters obtained from previous subtasks. "
        "Express the inradius as a fraction m/n in lowest terms, ensuring all algebraic manipulations are exact and no approximations are introduced prematurely. "
        "Confirm that the computed inradius corresponds to the inscribed circle of triangle ABC consistent with the problem's geometric setup and the chain of tangent circles."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2_3.content, answer2_3.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2_3, answer2_3], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, computing inradius, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 2.1: ", sub_tasks[-1])

    debate_instr_4_1 = (
        "Sub-task 3.1: Simplify the fraction representing the inradius to lowest terms, identify numerator m and denominator n, and compute the sum m + n as required by the problem. "
        "Verify the correctness of the simplification and the final sum. Provide the final answer along with a brief verification summary that confirms the logical consistency and correctness of all preceding steps."
    )
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_debate_4_1 = self.max_round
    all_thinking_4_1 = [[] for _ in range(N_debate_4_1)]
    all_answer_4_1 = [[] for _ in range(N_debate_4_1)]
    subtask_desc4_1 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": debate_instr_4_1,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_debate_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_4_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking_4_1[r-1] + all_answer_4_1[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instr_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying fraction and summing, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4_1[r].append(thinking4)
            all_answer_4_1[r].append(answer4)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4_1([taskInfo] + all_thinking_4_1[-1] + all_answer_4_1[-1], "Sub-task 3.1: Finalize fraction simplification and sum.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing fraction simplification and sum, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4_1)
    print("Step 3.1: ", sub_tasks[-1])

    reflect_inst_4_2 = (
        "Sub-task 3.2: Conduct a final reflexive review of the entire solution process, confirming that the geometric configuration, derivations, validations, and computations are consistent and free from the errors identified in previous attempts. "
        "Ensure that the assumptions made are justified, the formulas used are rigorously derived, and the final answer is reliable. "
        "This step serves as a safeguard against overlooked mistakes and reinforces confidence in the solution."
    )
    cot_reflect_instruction_4_2 = "Sub-task 3.2: Your problem is to review and confirm the entire solution process for correctness and consistency." + reflect_inst_4_2
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_2 = self.max_round
    cot_inputs_4_2 = [taskInfo, thinking0, answer0, thinking1, answer1, thinking2_1, answer2_1, thinking2_2, answer2_2, thinking2_3, answer2_3, thinking3, answer3, thinking4, answer4]
    subtask_desc4_2 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_reflect_instruction_4_2,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, reviewing solution process, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    for i in range(N_max_4_2):
        feedback, correct = await critic_agent_4_2([taskInfo, thinking4_2, answer4_2], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_2.extend([thinking4_2, answer4_2, feedback])
        thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_reflect_instruction_4_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, refining solution, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_2, answer4_2, sub_tasks, agents)
    return final_answer, logs
