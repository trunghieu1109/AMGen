async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Stage 0: Formal definitions and problem setup

    # Sub-task 1: Formal problem elements and event expression
    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem elements: specify the set of vertices of the regular octagon, "
        "the coloring function from vertices to {red, blue}, and the cyclic group of rotations of order 8 acting on the vertices. "
        "Precisely express the event that there exists a rotation mapping all blue vertices to positions originally colored red, "
        "using set and permutation notation. Avoid assumptions beyond the problem statement and clarify the role of the identity rotation explicitly."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formal problem definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Sub-task 2: Analyze rotation group C8 and cycle decompositions
    cot_instruction_2 = (
        "Sub-task 2: Analyze the structure of the rotation group C8 acting on the octagon vertices: enumerate all rotations, "
        "describe their cycle decompositions as permutations on vertex positions, and explain how these cycles affect the coloring sets. "
        "Emphasize the importance of cycle structure for counting colorings invariant or disjoint under rotations."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyze rotation group C8, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Sub-task 3: Define probabilistic model
    cot_instruction_3 = (
        "Sub-task 3: Formally define the probabilistic model: each vertex is independently colored red or blue with probability 1/2, "
        "resulting in a uniform probability space of size 2^8. Define the sample space and event space precisely, and clarify that all colorings are equally likely. "
        "Avoid conflating probability with counting at this stage."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, define probabilistic model, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Stage 1: Characterize colorings and count for each rotation

    # Sub-task 4: Characterize colorings satisfying rotational condition
    cot_instruction_4 = (
        "Sub-task 4: Characterize colorings satisfying the rotational condition: for a given rotation r, express the condition that the set of blue vertices is disjoint from its image under r (i.e., blue vertices map to originally red vertices). "
        "Formalize this condition in terms of the coloring and the rotation's action on vertex sets."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, characterize colorings for rotation disjointness, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Sub-task 5: For each rotation, count colorings B with B and r(B) disjoint
    cot_sc_instruction_5 = (
        "Sub-task 5: For each rotation r in C8, compute the number of colorings B such that B and r(B) are disjoint. "
        "Use the cycle decomposition of r to count these colorings by analyzing independent choices on cycles. "
        "Provide explicit formulas and examples for each rotation, ensuring clarity and correctness."
    )
    N_sc = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4, answer4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, count colorings disjoint under rotation, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_answers_5 + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent count of colorings disjoint under each rotation.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize counts for each rotation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    # Stage 1.1: Enumerate subgroups of C8

    # Sub-task 6_1: Enumerate all nontrivial subgroups of C8
    cot_instruction_6_1 = (
        "Sub-task 6_1: Enumerate all nontrivial subgroups of the cyclic group C8, including those generated by multiple rotations. "
        "List their elements explicitly and describe their structure to prepare for intersection analysis."
    )
    subtask_desc6_1 = {
        "subtask_id": "subtask_6_1",
        "instruction": cot_instruction_6_1,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "CoT"
    }
    thinking6_1, answer6_1 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction_6_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, enumerate subgroups of C8, thinking: {thinking6_1.content}; answer: {answer6_1.content}")
    sub_tasks.append(f"Sub-task 6_1 output: thinking - {thinking6_1.content}; answer - {answer6_1.content}")
    subtask_desc6_1['response'] = {"thinking": thinking6_1, "answer": answer6_1}
    logs.append(subtask_desc6_1)

    # Sub-task 6_2: For each subgroup H, compute number of colorings B disjoint under all non-identity rotations in H
    cot_sc_instruction_6_2 = (
        "Sub-task 6_2: For each nontrivial subgroup H of C8, compute the number of colorings B such that for every non-identity rotation r in H, B and r(B) are disjoint. "
        "Use the cycle decomposition principle applied to the subgroup action, carefully counting colorings disjoint under all rotations in H simultaneously. "
        "Provide detailed step-by-step calculations and formulas."
    )
    cot_agents_6_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6_2 = []
    possible_thinkings_6_2 = []
    subtask_desc6_2 = {
        "subtask_id": "subtask_6_2",
        "instruction": cot_sc_instruction_6_2,
        "context": ["user query", thinking6_1.content, answer6_1.content, thinking5, answer5],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6_2, answer6_2 = await cot_agents_6_2[i]([taskInfo, thinking6_1, answer6_1, thinking5, answer5], cot_sc_instruction_6_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6_2[i].id}, count colorings disjoint under subgroup, thinking: {thinking6_2.content}; answer: {answer6_2.content}")
        possible_answers_6_2.append(answer6_2)
        possible_thinkings_6_2.append(thinking6_2)
    final_decision_agent_6_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6_2, answer6_2 = await final_decision_agent_6_2([taskInfo] + possible_answers_6_2 + possible_thinkings_6_2, "Sub-task 6_2: Synthesize and choose the most consistent counts for subgroups.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesize counts for subgroups, thinking: {thinking6_2.content}; answer: {answer6_2.content}")
    sub_tasks.append(f"Sub-task 6_2 output: thinking - {thinking6_2.content}; answer - {answer6_2.content}")
    subtask_desc6_2['response'] = {"thinking": thinking6_2, "answer": answer6_2}
    logs.append(subtask_desc6_2)

    # Sub-task 6_3: Apply inclusion-exclusion over subgroups to compute total number of favorable colorings
    cot_sc_instruction_6_3 = (
        "Sub-task 6_3: Apply inclusion-exclusion over the subgroups of C8 to combine counts from subtask_6_2 and accurately compute the total number of colorings for which there exists at least one rotation satisfying the disjointness condition. "
        "Use Möbius inversion or explicit inclusion-exclusion formulas on the subgroup lattice. Provide clear, stepwise derivations and verify correctness of each term."
    )
    cot_agents_6_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6_3 = []
    possible_thinkings_6_3 = []
    subtask_desc6_3 = {
        "subtask_id": "subtask_6_3",
        "instruction": cot_sc_instruction_6_3,
        "context": ["user query", thinking6_2.content, answer6_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6_3, answer6_3 = await cot_agents_6_3[i]([taskInfo, thinking6_2, answer6_2], cot_sc_instruction_6_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6_3[i].id}, apply inclusion-exclusion, thinking: {thinking6_3.content}; answer: {answer6_3.content}")
        possible_answers_6_3.append(answer6_3)
        possible_thinkings_6_3.append(thinking6_3)
    final_decision_agent_6_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6_3, answer6_3 = await final_decision_agent_6_3([taskInfo] + possible_answers_6_3 + possible_thinkings_6_3, "Sub-task 6_3: Synthesize and finalize inclusion-exclusion count.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalize inclusion-exclusion count, thinking: {thinking6_3.content}; answer: {answer6_3.content}")
    sub_tasks.append(f"Sub-task 6_3 output: thinking - {thinking6_3.content}; answer - {answer6_3.content}")
    subtask_desc6_3['response'] = {"thinking": thinking6_3, "answer": answer6_3}
    logs.append(subtask_desc6_3)

    # Sub-task 6_4: Clarify treatment of identity rotation and edge cases
    cot_instruction_6_4 = (
        "Sub-task 6_4: Explicitly clarify the treatment of the identity rotation in the inclusion-exclusion process, ensuring that the empty coloring (all red) and other edge cases are correctly counted or excluded as appropriate. "
        "Document assumptions and reasoning to avoid ambiguity or miscounting."
    )
    subtask_desc6_4 = {
        "subtask_id": "subtask_6_4",
        "instruction": cot_instruction_6_4,
        "context": ["user query", thinking6_3.content, answer6_3.content],
        "agent_collaboration": "CoT"
    }
    thinking6_4, answer6_4 = await cot_agent([taskInfo, thinking6_3, answer6_3], cot_instruction_6_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, clarify identity rotation treatment, thinking: {thinking6_4.content}; answer: {answer6_4.content}")
    sub_tasks.append(f"Sub-task 6_4 output: thinking - {thinking6_4.content}; answer - {answer6_4.content}")
    subtask_desc6_4['response'] = {"thinking": thinking6_4, "answer": answer6_4}
    logs.append(subtask_desc6_4)

    # Stage 2: Calculate probability and verify

    # Sub-task 7: Calculate probability fraction m/n
    cot_sc_instruction_7 = (
        "Sub-task 7: Calculate the probability of the event as the ratio of the number of favorable colorings (from subtask_6_3) to the total number of colorings (2^8). "
        "Reduce the fraction to lowest terms, ensuring m and n are relatively prime positive integers. Provide the fraction m/n explicitly."
    )
    cot_agents_7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_7 = []
    possible_thinkings_7 = []
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_sc_instruction_7,
        "context": ["user query", thinking6_3.content, answer6_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking7, answer7 = await cot_agents_7[i]([taskInfo, thinking6_3, answer6_3], cot_sc_instruction_7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_7[i].id}, calculate probability fraction, thinking: {thinking7.content}; answer: {answer7.content}")
        possible_answers_7.append(answer7)
        possible_thinkings_7.append(thinking7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + possible_answers_7 + possible_thinkings_7, "Sub-task 7: Synthesize and finalize probability fraction.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalize probability fraction, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    # Sub-task 8: Verify inclusion-exclusion count by brute-force enumeration
    debate_instruction_8 = (
        "Sub-task 8: Verify the correctness of the inclusion-exclusion count and the resulting probability fraction by implementing a brute-force enumeration of all 256 colorings. "
        "Check that the count of colorings satisfying the rotational condition matches the inclusion-exclusion result exactly. "
        "Use this verification to identify and correct any discrepancies before finalizing the answer. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8 = self.max_round
    all_thinking8 = [[] for _ in range(N_max_8)]
    all_answer8 = [[] for _ in range(N_max_8)]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": debate_instruction_8,
        "context": ["user query", thinking6_3.content, answer6_3.content, thinking7.content, answer7.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_8):
        for i, agent in enumerate(debate_agents_8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking6_3, answer6_3, thinking7, answer7], debate_instruction_8, r, is_sub_task=True)
            else:
                input_infos_8 = [taskInfo, thinking6_3, answer6_3, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(input_infos_8, debate_instruction_8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verify inclusion-exclusion by brute-force, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Final verification and correction if needed.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalize verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)

    # Stage 3: Final confirmation and answer

    # Sub-task 9: Confirm fraction is reduced and summarize final answer
    debate_instruction_9 = (
        "Sub-task 9: Confirm that the fraction m/n is in lowest terms and that the verification step supports the correctness of the final probability. "
        "Summarize the reasoning steps, the final reduced fraction, and compute m+n as the problem requests. "
        "Present a clear, concise final answer with justification. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": debate_instruction_9,
        "context": ["user query", thinking7.content, answer7.content, thinking8.content, answer8.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                thinking9, answer9 = await agent([taskInfo, thinking7, answer7, thinking8, answer8], debate_instruction_9, r, is_sub_task=True)
            else:
                input_infos_9 = [taskInfo, thinking7, answer7, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, finalize answer, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Final confirmation and answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, final answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs
