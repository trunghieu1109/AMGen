async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction_0_1 = "Sub-task 0_1: Extract and list the key molecular biology features asserted in choice 1: (a) SARS-CoV-2 uses programmed -1 ribosomal frameshifting near the 5′ end; (b) the frameshift involves slipping by 1 nucleotide via slippery sequence and a pseudoknot; (c) the SARS-CoV-2 frameshifting pseudoknot conformation is mostly the same as SARS-CoV. Input: full text of choice 1."
    cot_agent_0_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting features of choice1, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    print("Subtask 0_1 answer: ", sub_tasks[-1])

    cot_instruction_0_2 = "Sub-task 0_2: Extract and list the key features asserted in choice 2: (a) SARS-CoV-2 nsp10/nsp14 form a heterodimer in mismatch repair; (b) the N-terminal ExoN domain of nsp14 binds nsp10 to form an active exonuclease that prevents dsRNA breakdown. Input: full text of choice 2."
    cot_agent_0_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_2, answer0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, extracting features of choice2, thinking: {thinking0_2.content}; answer: {answer0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    print("Subtask 0_2 answer: ", sub_tasks[-1])

    cot_instruction_0_3 = "Sub-task 0_3: Extract and list the key features asserted in choice 3: (a) ORF3a triggers caspase-8 activation/cleavage without altering Bcl-2 levels; (b) caspase-8 is part of the extrinsic apoptotic pathway; (c) Bcl-2 initiates the mitochondrial pathway; (d) thus ORF3a induces apoptosis via the extrinsic pathway. Input: full text of choice 3."
    cot_agent_0_3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_3, answer0_3 = await cot_agent_0_3([taskInfo], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, extracting features of choice3, thinking: {thinking0_3.content}; answer: {answer0_3.content}")
    sub_tasks.append(f"Sub-task 0_3 output: thinking - {thinking0_3.content}; answer - {answer0_3.content}")
    print("Subtask 0_3 answer: ", sub_tasks[-1])

    cot_instruction_0_4 = "Sub-task 0_4: Extract and list the key features asserted in choice 4: (a) in vitro frameshifting rate correlates linearly with the number of pseudoknot conformations; (b) both SARS-CoV and SARS-CoV-2 -1 frameshift signals show two conformations under tension; (c) these resemble pseudoknots from other frameshifting viruses. Input: full text of choice 4."
    cot_agent_0_4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking0_4, answer0_4 = await cot_agent_0_4([taskInfo], cot_instruction_0_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_4.id}, extracting features of choice4, thinking: {thinking0_4.content}; answer: {answer0_4.content}")
    sub_tasks.append(f"Sub-task 0_4 output: thinking - {thinking0_4.content}; answer - {answer0_4.content}")
    print("Subtask 0_4 answer: ", sub_tasks[-1])

    N = self.max_sc
    possible_answers1 = []
    thinkingmapping1 = {}
    answermapping1 = {}
    cot_sc_instruction_1_1 = "Sub-task 1_1: Gather and summarize experimental and structural evidence on SARS-CoV-2 programmed -1 ribosomal frameshifting: location, slippery sequence, pseudoknot structure, and comparison to SARS-CoV. Assess if choice1 assertions are supported."
    cot_sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i, agent in enumerate(cot_sc_agents1):
        thinking1_1, answer1_1 = await agent([taskInfo, thinking0_1, answer0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, evidence for frameshifting, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers1.append(answer1_1.content)
        thinkingmapping1[answer1_1.content] = thinking1_1
        answermapping1[answer1_1.content] = answer1_1
    final_answer1 = Counter(possible_answers1).most_common(1)[0][0]
    final_thinking1 = thinkingmapping1[final_answer1]
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {final_thinking1.content}; answer - {final_answer1}")
    print("Subtask 1_1 answer: ", sub_tasks[-1])

    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    cot_sc_instruction_1_2 = "Sub-task 1_2: Gather and summarize biochemical and structural data on the SARS-CoV-2 nsp10/nsp14 exonuclease complex: formation of heterodimer, role in mismatch repair, N-terminal ExoN binding, and dsRNA stability. Assess choice2 accuracy."
    cot_sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i, agent in enumerate(cot_sc_agents2):
        thinking1_2, answer1_2 = await agent([taskInfo, thinking0_2, answer0_2], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, evidence for exonuclease complex, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers2.append(answer1_2.content)
        thinkingmapping2[answer1_2.content] = thinking1_2
        answermapping2[answer1_2.content] = answer1_2
    final_answer2 = Counter(possible_answers2).most_common(1)[0][0]
    final_thinking2 = thinkingmapping2[final_answer2]
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {final_thinking2.content}; answer - {final_answer2}")
    print("Subtask 1_2 answer: ", sub_tasks[-1])

    possible_answers3 = []
    thinkingmapping3 = {}
    answermapping3 = {}
    cot_sc_instruction_1_3 = "Sub-task 1_3: Gather and summarize studies on SARS-CoV-2 ORF3a: effects on caspase-8 activation, Bcl-2 expression, and pathway of apoptosis induction. Assess choice3 accuracy."
    cot_sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i, agent in enumerate(cot_sc_agents3):
        thinking1_3, answer1_3 = await agent([taskInfo, thinking0_3, answer0_3], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, evidence for ORF3a apoptosis, thinking: {thinking1_3.content}; answer: {answer1_3.content}")
        possible_answers3.append(answer1_3.content)
        thinkingmapping3[answer1_3.content] = thinking1_3
        answermapping3[answer1_3.content] = answer1_3
    final_answer3 = Counter(possible_answers3).most_common(1)[0][0]
    final_thinking3 = thinkingmapping3[final_answer3]
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {final_thinking3.content}; answer - {final_answer3}")
    print("Subtask 1_3 answer: ", sub_tasks[-1])

    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    cot_sc_instruction_1_4 = "Sub-task 1_4: Gather and summarize single-molecule and biochemical studies of frameshifting pseudoknots under mechanical tension: conformations and frameshifting rates for SARS-CoV and SARS-CoV-2 compared to other viruses. Assess choice4 accuracy."
    cot_sc_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i, agent in enumerate(cot_sc_agents4):
        thinking1_4, answer1_4 = await agent([taskInfo, thinking0_4, answer0_4], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, evidence for pseudoknot conformations, thinking: {thinking1_4.content}; answer: {answer1_4.content}")
        possible_answers4.append(answer1_4.content)
        thinkingmapping4[answer1_4.content] = thinking1_4
        answermapping4[answer1_4.content] = answer1_4
    final_answer4 = Counter(possible_answers4).most_common(1)[0][0]
    final_thinking4 = thinkingmapping4[final_answer4]
    sub_tasks.append(f"Sub-task 1_4 output: thinking - {final_thinking4.content}; answer - {final_answer4}")
    print("Subtask 1_4 answer: ", sub_tasks[-1])

    cot_reflect_instruction_2_1 = "Sub-task 2_1: Compare extracted features from stage 0 with evidence from stage 1 for each choice. Decide if each choice's assertions conform to known data (True) or contradict (False). Output mapping choice number to True/False."
    cot_agent_2_1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_1 = [taskInfo, thinking0_1, answer0_1, final_thinking1, final_answer1, thinking0_2, answer0_2, final_thinking2, final_answer2, thinking0_3, answer0_3, final_thinking3, final_answer3, thinking0_4, answer0_4, final_thinking4, final_answer4]
    thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, classifying choices, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(self.max_round):
        feedback2_1, correct2_1 = await critic_agent_2_1([taskInfo, thinking2_1, answer2_1], "Critically evaluate the mapping classification for completeness and correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback2_1.content}; correct: {correct2_1.content}")
        if correct2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking2_1, answer2_1, feedback2_1])
        thinking2_1, answer2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining classification, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    print("Subtask 2_1 answer: ", sub_tasks[-1])

    cot_instruction_2_2 = "Sub-task 2_2: From the classification mapping in subtask 2_1, identify the single choice that is False and return that choice's number and text."
    cot_agent_2_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, identifying false choice, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    print("Subtask 2_2 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_2, answer2_2, sub_tasks, agents)
    return final_answer