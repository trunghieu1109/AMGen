async def forward_2(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize problem setting and event

    # Subtask 1: Define vertices, coloring, rotation group (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the problem setting by labeling the octagon vertices from 0 to 7, "
        "describing the coloring as a function from vertices to {red, blue}, and specifying the rotation group as the cyclic group of order 8 acting on vertex indices modulo 8. "
        "Emphasize that rotations are only multiples of 45° and that each vertex is independently colored red or blue with probability 1/2."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem setting, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Express event "exists rotation g such that g(blue) subset of red" (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Express the event 'there exists a rotation mapping all blue vertices to originally red vertices' in precise mathematical terms. "
        "Clarify that for some rotation g, the image of the blue set under g is a subset of the red set, and formalize this condition without attempting to solve it yet."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, formalizing event condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Describe probability space of colorings (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Describe the probability space of colorings: each of the 8 vertices is independently colored red or blue with probability 1/2, "
        "so the total number of colorings is 2^8 = 256, each equally likely with probability 1/256. Avoid any counting of favorable colorings at this stage."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, describing probability space, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Stage 0 Subtask 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Analyze rotations, cycle structures, and count compatible colorings

    # Subtask 1: List all rotations and their cycle structures (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify and list all rotations in the group (rotations by 0°, 45°, 90°, ..., 315°) and characterize their cycle structures on the vertex set. "
        "For each rotation, describe how it permutes the vertices and determine the lengths and number of cycles in its cycle decomposition."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, listing rotations and cycles, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize cycle structures of rotations.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Handle identity rotation special case (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Explicitly handle the identity rotation (0°) as a special case. "
        "Determine the condition on the coloring for the identity rotation to satisfy the event: since the blue set must map into the red set under identity, deduce that the blue set must be empty. "
        "Conclude that exactly one coloring (all red) satisfies this for the identity rotation."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_2.content, thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_0_2, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, handling identity rotation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize identity rotation special case.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: For each non-identity rotation, translate condition into coloring constraints (SC_CoT)
    cot_sc_instruction_1_3 = (
        "Sub-task 3: For each non-identity rotation, translate the condition that the blue set maps into the red set under that rotation into constraints on the coloring pattern. "
        "Use the cycle decomposition from subtask_1 to characterize these constraints, focusing on how colors must be assigned consistently on cycles to satisfy the condition."
    )
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_0_2.content, thinking_1_1.content, thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_0_2, thinking_1_1, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, translating rotation constraints, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize coloring constraints for non-identity rotations.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Subtask 4: Count number of colorings compatible with each rotation (SC_CoT)
    cot_sc_instruction_1_4 = (
        "Sub-task 4: For each rotation (including identity), count the number of colorings compatible with the rotation's condition derived in previous subtasks. "
        "Use the cycle structure and coloring constraints to compute |A_k|, the size of the set of colorings compatible with rotation k. "
        "Ensure the identity case count is exactly one as established."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_1.content, thinking_1_2.content, thinking_1_3.content, thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_1_4, answer_1_4 = await cot_agents_1_4[i]([taskInfo, thinking_1_1, thinking_1_2, thinking_1_3, thinking_0_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, counting compatible colorings, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
        possible_answers_1_4.append(answer_1_4)
        possible_thinkings_1_4.append(thinking_1_4)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 4: Synthesize counts of compatible colorings per rotation.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Subtask 5: Apply inclusion-exclusion to compute union size of favorable colorings (Debate)
    debate_instruction_1_5 = (
        "Sub-task 5: Apply the principle of inclusion-exclusion to compute the size of the union of all sets A_k (colorings compatible with at least one rotation). "
        "Enumerate all intersections |A_i ∩ A_j|, |A_i ∩ A_j ∩ A_l|, etc., using the counts from subtask_4 and the cycle structures to accurately find |⋃A_k|. "
        "Emphasize that this union size corresponds to the number of favorable colorings for the event, correcting the previous misuse of Burnside's lemma. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_5 = self.max_round
    all_thinking_1_5 = [[] for _ in range(N_max_1_5)]
    all_answer_1_5 = [[] for _ in range(N_max_1_5)]
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": debate_instruction_1_5,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_5):
        for i, agent in enumerate(debate_agents_1_5):
            if r == 0:
                thinking_1_5, answer_1_5 = await agent([taskInfo, thinking_1_4], debate_instruction_1_5, r, is_sub_task=True)
            else:
                input_infos_1_5 = [taskInfo, thinking_1_4] + all_thinking_1_5[r-1]
                thinking_1_5, answer_1_5 = await agent(input_infos_1_5, debate_instruction_1_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, inclusion-exclusion union size, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
            all_thinking_1_5[r].append(thinking_1_5)
            all_answer_1_5[r].append(answer_1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await final_decision_agent_1_5([taskInfo] + all_thinking_1_5[-1], "Sub-task 5: Finalize union size of favorable colorings.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 1.5: ", sub_tasks[-1])

    # Subtask 6: Validate union size logical consistency (Reflexion)
    reflect_inst_1_6 = (
        "Sub-task 6: Validate the computed union size from inclusion-exclusion by checking logical consistency: confirm that the union size is at least 1 (due to identity) and at most 256 (total colorings). "
        "Flag any inconsistencies or impossible values to prevent propagation of errors. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    critic_inst_1_6 = (
        "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    cot_agent_1_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_6 = [taskInfo, thinking_1_5]
    subtask_desc_1_6 = {
        "subtask_id": "stage_1.subtask_6",
        "instruction": reflect_inst_1_6,
        "context": ["user query", thinking_1_5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_6, answer_1_6 = await cot_agent_1_6(cot_inputs_1_6, reflect_inst_1_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_6.id}, validating union size, thinking: {thinking_1_6.content}; answer: {answer_1_6.content}")
    for i in range(self.max_round):
        feedback_1_6, correct_1_6 = await critic_agent_1_6([taskInfo, thinking_1_6], critic_inst_1_6, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_6.id}, feedback on union size validation, thinking: {feedback_1_6.content}; answer: {correct_1_6.content}")
        if correct_1_6.content == "True":
            break
        cot_inputs_1_6.extend([thinking_1_6, feedback_1_6])
        thinking_1_6, answer_1_6 = await cot_agent_1_6(cot_inputs_1_6, reflect_inst_1_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_6.id}, refining validation, thinking: {thinking_1_6.content}; answer: {answer_1_6.content}")
    sub_tasks.append(f"Stage 1 Subtask 6 output: thinking - {thinking_1_6.content}; answer - {answer_1_6.content}")
    subtask_desc_1_6['response'] = {"thinking": thinking_1_6, "answer": answer_1_6}
    logs.append(subtask_desc_1_6)
    print("Step 1.6: ", sub_tasks[-1])

    # Stage 2: Compute probability and final numeric result

    # Subtask 1: Express probability as simplified fraction (Reflexion)
    reflect_inst_2_1 = (
        "Sub-task 1: Express the probability of the event as the ratio of the union size of favorable colorings (from stage_1.subtask_5) to the total number of colorings (256). "
        "Simplify this fraction to lowest terms, ensuring numerator and denominator are coprime. Avoid any assumptions or shortcuts that could lead to incorrect simplification. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    critic_inst_2_1 = (
        "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking_1_5, thinking_1_6]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_inst_2_1,
        "context": ["user query", thinking_1_5.content, thinking_1_6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_inst_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, simplifying probability fraction, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], critic_inst_2_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback on fraction simplification, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_inst_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining fraction simplification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Compute m + n from simplified fraction (CoT)
    cot_instruction_2_2 = (
        "Sub-task 2: Compute the sum m + n, where m/n is the simplified fraction representing the probability. "
        "Prepare this final numeric result for presentation, ensuring clarity and correctness."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, computing m+n, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Subtask 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Stage 3: Final verification and presentation

    # Subtask 1: Aggregate and verify final answer (Reflexion)
    reflect_inst_3_1 = (
        "Sub-task 1: Aggregate and verify the final answer m + n against the problem statement. "
        "Confirm that all previous steps are consistent and that the final result logically follows from the computations. "
        "Prepare a clear, concise presentation of the answer."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": reflect_inst_3_1,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_2], reflect_inst_3_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_1.id}, final verification, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Stage 3 Subtask 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
