async def forward_17(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze and simplify the polynomial constraint a^2b + a^2c + b^2a + b^2c + c^2a + c^2b = 6,000,000 "
        "using the condition a + b + c = 300. Express the polynomial in terms of symmetric sums q = ab + bc + ca and r = abc, "
        "and derive the simplified relation r = 100q - 2,000,000 to facilitate enumeration."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, simplifying polynomial constraint, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Enumerate all triples (a,b,c) of nonnegative integers with a + b + c = 300 that satisfy the polynomial constraint. "
        "Explicitly split into two exhaustive cases: (1) degenerate case c = 100, where the constraint reduces to an identity for all (a,b) with a + b = 200, "
        "yielding a large family of solutions; (2) non-degenerate case c != 100, solve for ab = (2,000,000 - 100c(300 - c)) / (100 - c) and check all integer pairs (a,b) with a + b = 300 - c. "
        "Include zeros, equalities, and distinct values in the search. Use multiple agents with SC-CoT and Debate collaboration to ensure thoroughness and avoid premature pruning."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "SC_CoT | Debate"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, enumerating triples (a,b,c), thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_reflex_instruction_3 = (
        "Sub-task 3: Verify each candidate triple (a,b,c) obtained from enumeration rigorously against the original polynomial constraint to ensure exact satisfaction. "
        "Discard any false positives. Use programmatic or numeric checks to guarantee correctness. Employ SC-CoT and Reflexion collaboration to iteratively refine verification."
    )
    N_sc3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3) for _ in range(N_sc3)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_reflex_instruction_3,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    for i in range(N_sc3):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_reflex_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, verifying candidate triples, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    debate_reflex_instruction_4 = (
        "Sub-task 4: Iteratively refine the enumeration and verification process by analyzing missed cases or inconsistencies found during verification. "
        "Expand the candidate solution set accordingly and re-verify until no new valid solutions are found. Use Debate and Reflexion collaboration to ensure completeness and correctness."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_rounds_4)]
    all_answer_4 = [[] for _ in range(N_rounds_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_reflex_instruction_4,
        "context": ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"],
        "agent_collaboration": "Debate | Reflexion"
    }
    for r in range(N_rounds_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_reflex_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_reflex_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, refining candidate solutions, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)
    thinking4 = all_thinking_4[-1][0]
    answer4 = all_answer_4[-1][0]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc_4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_reflex_agg_instruction_5 = (
        "Stage 2 Sub-task 1: Aggregate the fully verified solution triples from Stage 1, carefully counting all distinct permutations of (a,b,c) "
        "while applying inclusion-exclusion principles to avoid double counting identical solutions arising from symmetry. Provide the final total count of valid triples "
        "and a comprehensive verification report confirming correctness and completeness. Use CoT, SC-CoT, and Reflexion collaboration."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_5 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_reflex_agg_instruction_5,
        "context": ["user query", "thinking of stage_1.subtask_4", "answer of stage_1.subtask_4"],
        "agent_collaboration": "CoT | SC_CoT | Reflexion"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_sc_reflex_agg_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, aggregating and counting valid triples, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
