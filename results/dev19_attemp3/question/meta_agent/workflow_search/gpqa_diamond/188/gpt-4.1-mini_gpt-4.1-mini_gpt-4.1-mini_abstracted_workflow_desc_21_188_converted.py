async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Explicitly define and clarify the concept of spontaneous symmetry breaking (SSB) in the context of the problem, including: "
        "(a) the nature of SSB in crystal formation and its implication that phonons are Nambu-Goldstone bosons arising from spontaneously broken continuous translational symmetry; "
        "(b) the distinction between Nambu-Goldstone bosons and topological solitons or defects; "
        "(c) precise criteria for what it means for an effective particle to be 'associated with spontaneously-broken symmetry.' "
        "This subtask addresses the previous failure to recognize phonons as arising from SSB and the ambiguity around Skyrmions by establishing a rigorous conceptual foundation for subsequent classification."
    )
    N_sc_0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, defining SSB concepts, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and finalize the rigorous definition and clarification of spontaneous symmetry breaking and related concepts.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1: Classify each effective particle (Magnon, Pion, Phonon, Skyrmion) into categories based on their physical origin and relation to spontaneous symmetry breaking, explicitly distinguishing: "
        "(a) Nambu-Goldstone bosons arising directly from SSB (e.g., Magnons, Pions, Phonons); "
        "(b) topological solitons or defects existing in broken-symmetry phases but not themselves Goldstone modes (e.g., Skyrmions); "
        "(c) excitations arising from explicit symmetry breaking if any. "
        "This classification must incorporate the clarified definitions from stage_0.subtask_1 to avoid previous misclassifications. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking0, answer0] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking1_1, answer1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, classifying particles, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_1_1[r].append(thinking1_1)
            all_answer_1_1[r].append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + all_thinking_1_1[-1] + all_answer_1_1[-1], "Sub-task 1: Synthesize and finalize classification of particles based on SSB.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Conduct a Reflexion-based critical review of the initial classification results, challenging assumptions and definitions especially regarding ambiguous cases like Skyrmions and phonons. "
        "This subtask aims to identify and correct any residual conceptual errors or ambiguities, ensuring that the classification aligns with rigorous physical understanding and the clarified problem framing. "
        "It also integrates relevant context from the detailed analysis and previous feedback to refine the classification. "
        + reflect_inst_1_2
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking0, answer0, thinking1_1, answer1_1]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking0.content, answer0.content, thinking1_1.content, answer1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking1_2, answer1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, reviewing classification, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    critic_inst_1_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_1_2):
        feedback1_2, correct1_2 = await critic_agent_1_2([taskInfo, thinking1_2, answer1_2], "Please review and provide the limitations of provided solutions." + critic_inst_1_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback1_2.content}; answer: {correct1_2.content}")
        if correct1_2.content.strip() == "True":
            break
        cot_inputs_1_2.extend([thinking1_2, answer1_2, feedback1_2])
        thinking1_2, answer1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining classification, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = (
        "Sub-task 3: Synthesize the refined classification and critical review outcomes to produce a final, coherent categorization of each particle's association with spontaneously-broken symmetry. "
        "This includes explicitly stating which particles are Nambu-Goldstone bosons, which are topological solitons, and which (if any) arise from explicit symmetry breaking, with clear justifications referencing the definitions and critical review."
    )
    N_sc_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking0.content, answer0.content, thinking1_1.content, answer1_1.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_3):
        thinking1_3, answer1_3 = await cot_agents_1_3[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, synthesizing final classification, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
        possible_answers_1_3.append(answer1_3)
        possible_thinkings_1_3.append(thinking1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_3, answer1_3 = await final_decision_agent_1_3([taskInfo, thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2] + possible_thinkings_1_3 + possible_answers_1_3, "Sub-task 3: Finalize the coherent categorization of particles with justifications.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_3.content}; answer - {answer1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking1_3, "answer": answer1_3}
    logs.append(subtask_desc_1_3)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Apply the refined and rigorously justified classification to determine which effective particle among the given options is NOT associated with a spontaneously-broken symmetry, providing a clear, well-supported explanation. "
        "This final determination must explicitly reference the distinctions established in previous subtasks and avoid the conceptual errors identified in prior attempts."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking1_3.content, answer1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking2_1, answer2_1 = await cot_agents_2_1[i]([taskInfo, thinking1_3, answer1_3], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, determining particle not associated with SSB, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        possible_answers_2_1.append(answer2_1)
        possible_thinkings_2_1.append(thinking2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking1_3, answer1_3] + possible_thinkings_2_1 + possible_answers_2_1, "Sub-task 1: Final determination of the particle not associated with spontaneously-broken symmetry.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
