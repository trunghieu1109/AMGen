async def forward_74(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    canonical_HA_aa = "YPYDVPDYA"
    canonical_HA_nt = "TACCCATACGATGTTCCAGATTACG"  
    stage_1a_instruction = "Sub-task 1a: Identify the start codon (ATG) in the provided nucleotide sequence and confirm the correct reading frame for translation of the recombinant protein construct."
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": stage_1a_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], stage_1a_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, identifying start codon and reading frame, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    stage_1b_instruction = "Sub-task 1b: Explicitly verify the presence and sequence integrity of the influenza hemagglutinin (HA) antigenic determinant nucleotide sequence at the N-terminus by aligning the translated amino acid sequence against the canonical HA tag amino acid sequence (YPYDVPDYA). Use the canonical HA tag amino acid sequence for alignment."
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": stage_1b_instruction,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "canonical HA tag amino acid sequence: YPYDVPDYA"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a, canonical_HA_aa], stage_1b_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, verifying HA tag presence and integrity, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    stage_2_instruction_2 = "Sub-task 2: Translate the entire nucleotide sequence in the confirmed reading frame into the corresponding amino acid sequence using a computational codon-to-amino acid translation module, and detect any premature in-frame stop codons (TAA, TAG, TGA) within the HA tag or adjacent regions."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": stage_2_instruction_2,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1b, answer_1b], stage_2_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, translating nucleotide sequence and detecting premature stop codons, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2.content)
        thinkingmapping_2[answer_2.content] = thinking_2
        answermapping_2[answer_2.content] = answer_2
    answer_2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking_2 = thinkingmapping_2[answer_2_content]
    answer_2 = answermapping_2[answer_2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])
    reflexion_instruction = "Reflexion Checkpoint: Review and cross-validate the translation and premature stop codon detection results from Sub-task 2 to ensure accuracy before proceeding."
    cot_agent_reflexion = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_reflexion = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflexion = self.max_round
    reflexion_inputs = [taskInfo, thinking_2, answer_2]
    subtask_desc_reflexion = {
        "subtask_id": "reflexion_checkpoint",
        "instruction": reflexion_instruction,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_reflexion, answer_reflexion = await cot_agent_reflexion(reflexion_inputs, reflexion_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, reviewing translation and stop codon detection, thinking: {thinking_reflexion.content}; answer: {answer_reflexion.content}")
    for i in range(N_max_reflexion):
        feedback, correct = await critic_agent_reflexion([taskInfo, thinking_reflexion, answer_reflexion], "Please review the translation and stop codon detection for accuracy and completeness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_reflexion.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        reflexion_inputs.extend([thinking_reflexion, answer_reflexion, feedback])
        thinking_reflexion, answer_reflexion = await cot_agent_reflexion(reflexion_inputs, reflexion_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflexion.id}, refining review, thinking: {thinking_reflexion.content}; answer: {answer_reflexion.content}")
    sub_tasks.append(f"Reflexion checkpoint output: thinking - {thinking_reflexion.content}; answer - {answer_reflexion.content}")
    subtask_desc_reflexion['response'] = {"thinking": thinking_reflexion, "answer": answer_reflexion}
    logs.append(subtask_desc_reflexion)
    print("Reflexion checkpoint: ", sub_tasks[-1])
    stage_3_instruction_4 = "Sub-task 4: Compare the translated amino acid sequence of the HA tag region with the canonical HA tag sequence (YPYDVPDYA) to identify any missense mutations or sequence deviations that could affect protein expression or function, based on the reflexion checkpoint output."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": stage_3_instruction_4,
        "context": ["user query", "thinking of reflexion checkpoint", "answer of reflexion checkpoint"],
        "agent_collaboration": "CoT"
    }
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_reflexion, answer_reflexion], stage_3_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, comparing HA tag amino acid sequence with canonical sequence, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Evaluate the potential impact of the absence of a linker sequence between the HA tag and the GADD45G coding sequence on protein stability, including the risk of proteolysis of the nascent chain, based on Sub-task 4 output."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating linker absence impact, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking5[r].append(thinking_5)
            all_answer5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on proteolysis risk due to linker absence.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding on proteolysis risk, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    stage_3_instruction_6 = "Sub-task 6: Assess the biological plausibility of translation termination caused by the presence of a UAA stop codon and the non-existence of a corresponding tRNA in the mouse system, to rule out or confirm this as a cause of failed protein overexpression, based on the reflexion checkpoint output."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": stage_3_instruction_6,
        "context": ["user query", "thinking of reflexion checkpoint", "answer of reflexion checkpoint"],
        "agent_collaboration": "CoT"
    }
    thinking_6, answer_6 = await cot_agent_6([taskInfo, thinking_reflexion, answer_reflexion], stage_3_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, assessing biological plausibility of UAA stop codon translation termination, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    stage_4_instruction_7 = "Sub-task 7: Integrate and synthesize findings from sequence verification, translation analysis, mutation detection, linker evaluation, and biological context to identify and clearly distinguish the primary cause of the failure to overexpress the protein, prioritizing premature translation termination due to stop codons over secondary factors."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking_4, answer_4, thinking_5, answer_5, thinking_6, answer_6]
    subtask_desc_7 = {
        "subtask_id": "subtask_7",
        "instruction": stage_4_instruction_7,
        "context": ["user query", "thinking and answer of subtask 4", "thinking and answer of subtask 5", "thinking and answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking_7, answer_7 = await cot_agent_7(cot_inputs_7, stage_4_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, integrating all findings, thinking: {thinking_7.content}; answer: {answer_7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking_7, answer_7], "Please review the integration of findings and identify any inconsistencies or improvements.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking_7, answer_7, feedback])
        thinking_7, answer_7 = await cot_agent_7(cot_inputs_7, stage_4_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining integration, thinking: {thinking_7.content}; answer: {answer_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_7.content}; answer - {answer_7.content}")
    subtask_desc_7['response'] = {"thinking": thinking_7, "answer": answer_7}
    logs.append(subtask_desc_7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking_7, answer_7, sub_tasks, agents)
    return final_answer, logs
