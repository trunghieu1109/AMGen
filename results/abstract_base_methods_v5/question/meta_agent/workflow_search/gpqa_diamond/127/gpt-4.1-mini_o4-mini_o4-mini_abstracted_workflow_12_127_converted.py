async def forward_127(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Stage 1: Codon Optimization and Verification
    
    # Subtask 1a: Codon Optimization with Self-Consistency CoT
    cot_sc_instruction_1a = (
        "Subtask 1a: Reverse-translate the given Human P53 amino acid sequence into a nucleotide sequence optimized for expression in E. coli BL21, "
        "using the standard codon usage table for E. coli BL21. Generate multiple independent optimized sequences and reach consensus."
    )
    N_sc_1a = self.max_sc
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1a)]
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query", "Human P53 amino acid sequence", "E. coli BL21 codon usage table"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1a):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, codon optimizing Human P53, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    answer1a_content = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[answer1a_content]
    answer1a = answermapping_1a[answer1a_content]
    sub_tasks.append(f"Subtask 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    # Subtask 1b: Verification of optimized sequence against plasmid inserts with Reflexion
    cot_reflect_instruction_1b = (
        "Subtask 1b: Verify that the codon-optimized nucleotide sequence exactly matches one of the four provided plasmid sequences by performing rigorous nucleotide-level alignments, "
        "checking for frameshifts, mutations, and completeness. Review and refine the verification iteratively."
    )
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1b = self.max_round
    cot_inputs_1b = [taskInfo, thinking1a, answer1a]
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_reflect_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a", "4 plasmid sequences"],
        "agent_collaboration": "Reflexion"
    }
    thinking1b, answer1b = await cot_agent_1b(cot_inputs_1b, cot_reflect_instruction_1b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1b.id}, verifying optimized sequence against plasmids, thinking: {thinking1b.content}; answer: {answer1b.content}")
    for i in range(N_max_1b):
        feedback, correct = await critic_agent_1b([taskInfo, thinking1b, answer1b], "Please review the nucleotide-level alignment verification and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1b.extend([thinking1b, answer1b, feedback])
        thinking1b, answer1b = await cot_agent_1b(cot_inputs_1b, cot_reflect_instruction_1b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1b.id}, refining verification, thinking: {thinking1b.content}; answer: {answer1b.content}")
    sub_tasks.append(f"Subtask 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    # Stage 2: Alignment and Expression Suitability Evaluation
    
    # Subtask 2a: Alignment Agent with Self-Consistency CoT
    cot_sc_instruction_2a = (
        "Subtask 2a: Perform explicit nucleotide and amino acid sequence alignments between the validated ORF from Subtask 1b and each of the four plasmid sequences. "
        "Calculate alignment metrics including percent identity, presence of mutations, frameshifts, and premature stop codons. Generate multiple independent assessments and reach consensus."
    )
    N_sc_2a = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2a)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b", "4 plasmid sequences"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2a):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1b, answer1b], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, performing sequence alignments, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Subtask 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    # Subtask 2b: Expression-related evaluation with Reflexion and specialized agents
    cot_reflect_instruction_2b = (
        "Subtask 2b: Evaluate each plasmid's suitability for rapid expression and purification of Human P53 in E. coli BL21 by assessing: "
        "(a) sequence integrity and exact match to the intended coding sequence, "
        "(b) codon usage bias and presence of rare codons, "
        "(c) predicted mRNA secondary structure affecting translation efficiency, "
        "(d) presence and compatibility of ribosome binding sites and promoters, "
        "and (e) plasmid backbone features relevant to expression and stability. "
        "Use Reflexion to iteratively refine the evaluation integrating these factors."
    )
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2b = self.max_round
    cot_inputs_2b = [taskInfo, thinking2a, answer2a]
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_reflect_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "expression-related metadata"],
        "agent_collaboration": "Reflexion"
    }
    thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, evaluating plasmid expression suitability, thinking: {thinking2b.content}; answer: {answer2b.content}")
    for i in range(N_max_2b):
        feedback, correct = await critic_agent_2b([taskInfo, thinking2b, answer2b], "Please review the plasmid expression suitability evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2b.extend([thinking2b, answer2b, feedback])
        thinking2b, answer2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, refining plasmid expression evaluation, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Subtask 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    # Stage 3: Final Plasmid Selection and Cross-Validation
    
    debate_instruction_3 = (
        "Subtask 3: Integrate all sequence verification and expression-related evaluations from Subtasks 2a and 2b to select the plasmid that best ensures accurate, efficient, and rapid expression and purification of Human P53 in E. coli BL21. "
        "Perform a final cross-validation to confirm that the chosen plasmid's nucleotide sequence translates exactly to the Human P53 amino acid sequence and meets expression optimization criteria."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2a, answer2a, thinking2b, answer2b], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2a, answer2a, thinking2b, answer2b] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting plasmid, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Subtask 3: Make final decision on the best plasmid for Human P53 expression and cross-validate the choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on plasmid selection, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
