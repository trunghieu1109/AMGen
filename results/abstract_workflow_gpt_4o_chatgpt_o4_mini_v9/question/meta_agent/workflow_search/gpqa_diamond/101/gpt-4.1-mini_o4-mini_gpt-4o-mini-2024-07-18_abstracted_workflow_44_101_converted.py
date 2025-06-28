async def forward_101(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Analyze molecular features, vector design, experimental setup, and mouse breeding
    cot_instruction_1 = (
        "Sub-task 1: Analyze the molecular and structural features of the receptor and ligand proteins, "
        "including the receptor's transmembrane barrel with eight alpha helices and beta sheets, and the ligand's coiled-coil of two alpha helices, "
        "to understand their interaction potential and expression context in neural crest cells."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing molecular and structural features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1, analyze the design and components of the bicistronic lox-Cre vector, "
        "including the arrangement of ligand ORF fused to mCherry at 5', receptor ORF fused to eGFP at 3', the CBA promoter driving ligand expression, "
        "the IRES element upstream of the receptor, and the presence of loxP-stop-loxP and lox2272-stop-lox2272 cassettes between ORFs and fluorescent tags."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing vector design, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Sub-task 3: Analyze the experimental setup involving antibody generation against overexpressed proteins, "
        "Western blot validation of construct expression in primary astrocyte culture, and the use of actin as a loading control to confirm protein expression. "
        "Refine the analysis iteratively to ensure validity."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing experimental setup, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                                "Critically evaluate the experimental setup analysis for completeness and correctness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining experimental setup analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_reflect_instruction_4 = (
        "Sub-task 4: Analyze the mouse breeding strategy involving homozygous construct mice crossed with SOX10-Cre hemizygous mice, "
        "and the subsequent visualization of fluorescent signals under confocal microscopy to assess in vivo expression and recombination. "
        "Refine iteratively to ensure accuracy."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, analyzing mouse breeding and visualization, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                                "Critically evaluate the mouse breeding and fluorescence visualization analysis.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining mouse breeding analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Evaluate possible reasons for absence of green fluorescence
    # Sub-task 5: Evaluate frame maintenance of receptor-eGFP fusion
    cot_instruction_5 = (
        "Sub-task 5: Evaluate whether the bicistronic vector design allows proper expression and frame maintenance of the receptor-eGFP fusion protein, "
        "considering the presence of IRES, stop cassettes, and fusion junctions, to determine if the receptor and eGFP are in frame."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating frame maintenance, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Evaluate if absence of green signal is due to lack of Cre-mediated recombination
    cot_instruction_6 = (
        "Sub-task 6: Evaluate whether the absence of green fluorescence signal in the offspring is due to lack of Cre-mediated recombination removing the lox2272-stop-lox2272 cassette upstream of the receptor-eGFP ORF, "
        "considering the SOX10-Cre expression pattern and lox site specificity."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, evaluating Cre recombination, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Evaluate paracrine expression possibility
    cot_instruction_7 = (
        "Sub-task 7: Evaluate whether the ligand and receptor proteins are expressed in a paracrine manner rather than in the same cells, "
        "which could explain the absence of green fluorescence in cells expressing Cre recombinase."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking1, answer1, thinking4, answer4], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, evaluating paracrine expression, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Evaluate enhancer/promoter sufficiency
    cot_instruction_8 = (
        "Sub-task 8: Evaluate whether the enhancer or promoter elements driving ligand and receptor expression are missing or insufficient in the construct or mouse model, "
        "potentially causing lack of receptor-eGFP expression despite Cre presence."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking2, answer2, thinking4, answer4], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, evaluating enhancer/promoter sufficiency, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])

    # Sub-task 9: Evaluate receptor-eGFP trafficking and localization
    cot_instruction_9 = (
        "Sub-task 9: Evaluate whether the receptor-eGFP fusion protein is properly trafficked or if it is retained in intracellular compartments such as the Golgi apparatus, "
        "which could prevent detection of green fluorescence at the expected cellular location."
    )
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await cot_agent_9([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, evaluating protein trafficking, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])

    # Stage 2: Integrate findings to determine most likely explanation
    debate_instruction_10 = (
        "Sub-task 10: Integrate findings from frame evaluation (Sub-task 5), recombination efficiency (Sub-task 6), expression pattern (Sub-task 7), "
        "enhancer/promoter sufficiency (Sub-task 8), and protein localization (Sub-task 9) to determine the most likely explanation for the absence of green fluorescence signal in the SOX10-Cre offspring."
    )
    debate_agents_10 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_10 = self.max_round
    all_thinking10 = [[] for _ in range(N_max_10)]
    all_answer10 = [[] for _ in range(N_max_10)]
    for r in range(N_max_10):
        for i, agent in enumerate(debate_agents_10):
            if r == 0:
                input_infos_10 = [taskInfo, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9]
            else:
                input_infos_10 = [taskInfo, thinking5, answer5, thinking6, answer6, thinking7, answer7, thinking8, answer8, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
            thinking10, answer10 = await agent(input_infos_10, debate_instruction_10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating findings, thinking: {thinking10.content}; answer: {answer10.content}")
            all_thinking10[r].append(thinking10)
            all_answer10[r].append(answer10)
    final_decision_agent_10 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_decision_agent_10([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make final decision on the most likely explanation for absence of green fluorescence.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer
