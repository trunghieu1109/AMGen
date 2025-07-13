async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_N = self.max_sc

    # Stage 0: Setup and foundational analysis

    # Subtask 0_1: Define geometric setup (SC-CoT)
    cot_sc_instruction_0_1 = (
        "Subtask 0_1: Formally define the geometric setup of the problem by representing the regular dodecagon as 12 equally spaced points on a circle. "
        "Enumerate all sides and diagonals as chords connecting these vertices, specifying labeling and orientation conventions without assumptions beyond uniform spacing."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, defining geometric setup, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_answers_0_1 + possible_thinkings_0_1, "Subtask 0_1: Synthesize and choose the most consistent geometric setup definition.", is_sub_task=True)
    sub_tasks.append(f"Subtask 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Subtask 0_2: Define problem constraints (SC-CoT)
    cot_sc_instruction_0_2 = (
        "Subtask 0_2: Precisely define the problem constraints that rectangles must have all four sides lying on chords of the dodecagon (including all sides and diagonals). "
        "Clarify that all chords between vertices are considered, rectangles must be non-degenerate with positive area, and must be fully contained inside the polygon. "
        "Explicitly exclude degenerate or partially outside rectangles."
    )
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, defining problem constraints, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, "Subtask 0_2: Synthesize and choose the most consistent problem constraints definition.", is_sub_task=True)
    sub_tasks.append(f"Subtask 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Subtask 0_3: Analyze geometric conditions for rectangles (SC-CoT)
    cot_sc_instruction_0_3 = (
        "Subtask 0_3: Analyze the geometric conditions for four vertices on the circle to form a rectangle with edges along polygon chords. "
        "Derive necessary and sufficient conditions on chord orientations and lengths, including the key insight that pairs of perpendicular chords correspond to chord skips of d and 6-d vertices (for d=1,2,3), reflecting 90° arc differences. "
        "Avoid restricting to only (3,6) chord pairs."
    )
    cot_sc_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_3[i]([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_3[i].id}, analyzing rectangle conditions, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_answers_0_3 + possible_thinkings_0_3, "Subtask 0_3: Synthesize and choose the most consistent rectangle condition analysis.", is_sub_task=True)
    sub_tasks.append(f"Subtask 0_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Subtask 0_4: Formalize symmetries (SC-CoT)
    cot_sc_instruction_0_4 = (
        "Subtask 0_4: Identify and formalize the symmetries of the regular dodecagon relevant to counting rectangles, including rotational symmetry of order 12 and reflection symmetries. "
        "Explicitly characterize how these symmetries affect vertex labeling and chord equivalences, and how they induce potential double counting in rectangle enumeration. "
        "Prepare this symmetry formalization as a context to be passed to all subsequent subtasks to ensure consistent counting."
    )
    cot_sc_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_sc_instruction_0_4,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_4[i]([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_4[i].id}, formalizing symmetries, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_4.append(answer)
        possible_thinkings_0_4.append(thinking)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + possible_answers_0_4 + possible_thinkings_0_4, "Subtask 0_4: Synthesize and choose the most consistent symmetry formalization.", is_sub_task=True)
    sub_tasks.append(f"Subtask 0_4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)

    # Stage 1: Enumeration and perpendicular chord identification

    # Subtask 1_1: Enumerate all chords and classify by skip number d (SC-CoT)
    cot_sc_instruction_1_1 = (
        "Subtask 1_1: Enumerate all chords (sides and diagonals) of the dodecagon, classifying them by their skip number d (number of vertices skipped) and corresponding geometric properties such as length and orientation. "
        "Explicitly list chords for d=1 to 5 (since d=6 is the diameter) and prepare this classification as input for perpendicular chord identification."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1, thinking_0_4, answer_0_4], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating chords, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Subtask 1_1: Synthesize and choose the most consistent chord enumeration.", is_sub_task=True)
    sub_tasks.append(f"Subtask 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Subtask 1_2: Identify all perpendicular chord pairs (SC-CoT)
    cot_sc_instruction_1_2 = (
        "Subtask 1_2: Develop and apply criteria to identify all pairs of chords that are perpendicular, based on the chord skip numbers (d, 6-d) for d=1,2,3. "
        "For each such pair, verify that the chords are indeed perpendicular by geometric reasoning or angle calculations. "
        "Avoid limiting to only one pair of skip numbers. Document all such perpendicular chord pairs as candidates for opposite sides of rectangles."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_1.content, answer_1_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_1_2[i]([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1, thinking_0_4, answer_0_4], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, identifying perpendicular chord pairs, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Subtask 1_2: Synthesize and choose the most consistent perpendicular chord pairs identification.", is_sub_task=True)
    sub_tasks.append(f"Subtask 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Subtask 1_3: Construct all rectangles from perpendicular chord pairs (SC-CoT)
    cot_sc_instruction_1_3 = (
        "Subtask 1_3: Combine the identified perpendicular chord pairs to construct all possible quadruples of vertices that form rectangles with edges on polygon chords. "
        "Ensure that each rectangle's four edges correspond to sides or diagonals of the dodecagon, and that the rectangle is non-degenerate and fully inside the polygon. "
        "Explicitly generate the list of all such rectangles, grouped by their chord skip families (d=1,2,3)."
    )
    cot_sc_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_1_3[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_3[i].id}, constructing rectangles, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_3.append(answer)
        possible_thinkings_1_3.append(thinking)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Subtask 1_3: Synthesize and choose the most consistent rectangle construction.", is_sub_task=True)
    sub_tasks.append(f"Subtask 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Subtask 1_4: Remove duplicates caused by symmetry (Debate)
    debate_instruction_1_4 = (
        "Subtask 1_4: Analyze the list of rectangles generated to identify duplicates caused by the polygon's rotational and reflection symmetries and vertex labeling. "
        "Develop a systematic method to remove duplicates and correct overcounting, using the symmetry formalization from stage_0.subtask_4. "
        "Provide a corrected list of unique rectangles and justify the counting adjustments with clear symmetry arguments. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_3, answer_1_3, thinking_0_4, answer_0_4], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, answer_1_3, thinking_0_4, answer_0_4] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, removing duplicates, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_4[r].append(thinking)
            all_answer_1_4[r].append(answer)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + all_thinking_1_4[-1] + all_answer_1_4[-1], "Subtask 1_4: Finalize unique rectangle list after symmetry deduplication.", is_sub_task=True)
    sub_tasks.append(f"Subtask 1_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2: Final counting and verification

    # Subtask 2_1: Count total distinct rectangles (Reflexion)
    reflect_instruction_2_1 = (
        "Subtask 2_1: Count the total number of distinct rectangles inside the regular dodecagon formed by sides or diagonals, using the corrected unique rectangle list from stage_1.subtask_4. "
        "Explicitly verify that the count includes all three rectangle families corresponding to chord skip pairs (1,5), (2,4), and (3,3), yielding the total of 36 rectangles. "
        "Provide detailed justification for the final count, referencing geometric and combinatorial reasoning. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_4, answer_1_4]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, counting rectangles, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining count, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Subtask 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Subtask 2_2: Cross-verify final count (Debate)
    debate_instruction_2_2 = (
        "Subtask 2_2: Cross-verify the final rectangle count by comparing results from different counting approaches or agents, resolving any discrepancies through discussion and consensus. "
        "Use the symmetry formalization and combinatorial arguments to adjudicate and confirm the correctness and completeness of the final count. "
        "Document the reasoning process and final agreed-upon answer clearly. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": debate_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_1, answer_2_1, thinking_0_4, answer_0_4], debate_instruction_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1, answer_2_1, thinking_0_4, answer_0_4] + all_thinking_2_2[r-1] + all_answer_2_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-verifying count, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_2[r].append(thinking)
            all_answer_2_2[r].append(answer)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + all_thinking_2_2[-1] + all_answer_2_2[-1], "Subtask 2_2: Finalize and confirm the total rectangle count.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing count, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Subtask 2_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
