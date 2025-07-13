async def forward_2(self, taskInfo):
    from collections import Counter
    import math
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate all 256 possible colorings of the octagon's 8 vertices, "
        "each vertex colored red or blue independently with probability 1/2. "
        "For each coloring, check if there exists a rotation (0 to 7 steps) such that the blue vertices after rotation are a subset of the original red vertices. "
        "Count the total number of such valid colorings. Provide the count explicitly as an integer. "
        "This brute-force enumeration serves as a ground-truth baseline."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, brute-force enumeration, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Formally define the validity condition for a coloring: there exists a rotation r (0 to 7) such that the set of blue vertices after rotation is a subset of the original red vertices. "
        "Express this condition rigorously using set notation and group actions (rotations as cyclic shifts on vertex indices). "
        "Represent colorings as 8-bit binary tuples (blue=1, red=0). Clarify that only rotations (no reflections) are considered. "
        "This formalization will underpin the combinatorial counting and inclusion-exclusion steps."
    )
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, formalized validity condition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent formal definition of valid colorings.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: For each rotation r in {0,...,7}, define the subset A_r of colorings where the blue vertices after rotation r are contained in the original red vertices. "
        "Compute the size |A_r| explicitly. Derive these counts by analyzing the constraints imposed by each rotation on the coloring pattern. "
        "Provide explicit numeric values for all |A_r|. Avoid unverified assumptions."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content, thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2, thinking_1_1], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, computed |A_r| for rotations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose consistent |A_r| counts.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Compute all pairwise intersections |A_i ∩ A_j| for distinct rotations i, j, and triple intersections |A_i ∩ A_j ∩ A_k| if nonempty. "
        "Use these to apply the inclusion-exclusion formula explicitly to find |⋃_r A_r|, the total number of valid colorings. "
        "Provide intermediate numeric results and verify no double counting occurs."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage2_subtask2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, computed intersections and inclusion-exclusion, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and finalize inclusion-exclusion union size.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_2_3 = (
        "Sub-task 3: Cross-validate the inclusion-exclusion union size |⋃_r A_r| obtained above with the brute-force count of valid colorings from stage1_subtask1. "
        "If discrepancies arise, trace and debug the inclusion-exclusion terms or brute-force enumeration. Confirm agreement between these two independent methods before proceeding."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage2_subtask3",
        "instruction": debate_instr_2_3,
        "context": ["user query", thinking_2_2.content, answer_2_2.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2, thinking_1_1, answer_1_1], debate_instr_2_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2, answer_2_2, thinking_1_1, answer_1_1] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validation, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_3[r].append(thinking_i)
            all_answer_2_3[r].append(answer_i)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1], "Sub-task 3: Finalize cross-validation of counts.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_3_1 = (
        "Sub-task 1: Using the verified count of valid colorings (numerator) and total colorings (256), "
        "compute the probability as a fraction m/n in lowest terms. Explicitly calculate the greatest common divisor (GCD) of numerator and denominator to simplify the fraction. "
        "Output the reduced fraction and the sum m + n as required by the problem."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage3_subtask1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_3.content, answer_2_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_3, answer_2_3], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, fraction simplification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction_3_2 = (
        "Sub-task 2: Perform a final verification of the fraction simplification and the computed sum m + n. "
        "Cross-check all numeric results, confirm that the fraction matches the probability derived from counting, and reflect on the entire reasoning process to ensure no logical or computational errors remain. "
        "Provide a final, justified answer ready for submission."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage3_subtask2",
        "instruction": cot_reflect_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, verifying final answer, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(N_max_3_2):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide the limitations or confirm correctness. Output 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining verification, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
