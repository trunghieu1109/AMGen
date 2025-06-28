async def forward_48(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the general mechanism and characteristics of sigmatropic rearrangements, focusing on the migration of terminal pi bonds into sigma bonds, and understand why these reactions are typically thermodynamically favored, with context from the query."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing sigmatropic rearrangements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Review the specific mechanistic features, reaction conditions, and examples of Cope and Claisen rearrangements as representative sigmatropic rearrangements to establish a mechanistic framework applicable to the given reactions, based on Sub-task 1 outputs."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, reviewing Cope and Claisen rearrangements, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3_1 = "Sub-task 3_1: For the first reaction (1,1-dimethoxyethan-1-amine + but-3-en-2-ol + H+ + Heat), identify and characterize the key reactive intermediates formed under acid catalysis, including iminium ion formation and possible enamine intermediates, based on Sub-task 2 outputs."
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3_1, answer3_1 = await cot_agent_3_1([taskInfo, thinking2, answer2], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, identifying reactive intermediates for first reaction, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {
        "thinking": thinking3_1,
        "answer": answer3_1
    }
    logs.append(subtask_desc3_1)
    print("Step 3_1: ", sub_tasks[-1])
    
    cot_instruction_3_2 = "Sub-task 3_2: Analyze the intramolecular cyclization pathways of the intermediates identified in Sub-task 3_1, focusing on acid-catalyzed ring closure mechanisms leading to heterocyclic products such as 3,4-dihydro-2H-pyran derivatives."
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking2, answer2, thinking3_1, answer3_1]
    subtask_desc3_2 = {
        "subtask_id": "subtask_3_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 3_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_2, answer3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, analyzing cyclization pathways, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    for i in range(N_max_3_2):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking3_2, answer3_2], "please review the cyclization pathway analysis and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, providing feedback on cyclization pathways, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking3_2, answer3_2, feedback])
        thinking3_2, answer3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining cyclization pathway analysis, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3_2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {
        "thinking": thinking3_2,
        "answer": answer3_2
    }
    logs.append(subtask_desc3_2)
    print("Step 3_2: ", sub_tasks[-1])
    
    cot_instruction_3_3 = "Sub-task 3_3: Predict the final product A of the first reaction by integrating the mechanistic insights from Sub-tasks 3_1 and 3_2, ensuring the product structure reflects acid-catalyzed cyclization rather than classical open-chain Claisen rearrangement."
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_3 = self.max_round
    cot_inputs_3_3 = [taskInfo, thinking2, answer2, thinking3_1, answer3_1, thinking3_2, answer3_2]
    subtask_desc3_3 = {
        "subtask_id": "subtask_3_3",
        "instruction": cot_instruction_3_3,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 3_1", "thinking and answer of subtask 3_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_instruction_3_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, predicting product A, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    for i in range(N_max_3_3):
        feedback, correct = await critic_agent_3_3([taskInfo, thinking3_3, answer3_3], "please review the product A prediction and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, providing feedback on product A prediction, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_3.extend([thinking3_3, answer3_3, feedback])
        thinking3_3, answer3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_instruction_3_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining product A prediction, thinking: {thinking3_3.content}; answer: {answer3_3.content}")
    sub_tasks.append(f"Sub-task 3_3 output: thinking - {thinking3_3.content}; answer - {answer3_3.content}")
    subtask_desc3_3['response'] = {
        "thinking": thinking3_3,
        "answer": answer3_3
    }
    logs.append(subtask_desc3_3)
    print("Step 3_3: ", sub_tasks[-1])
    
    cot_instruction_4_1 = "Sub-task 4_1: For the second reaction ((3R,4S)-3,4-dimethylhexa-1,5-diyne + Heat), list all plausible thermal pericyclic pathways, including Cope rearrangement, Bergman cyclization, and other diyne-specific cyclizations, supported by literature precedents, based on Sub-task 2 outputs."
    cot_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4_1 = {
        "subtask_id": "subtask_4_1",
        "instruction": cot_instruction_4_1,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking4_1, answer4_1 = await cot_agent_4_1([taskInfo, thinking2, answer2], cot_instruction_4_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4_1.id}, listing plausible pericyclic pathways for second reaction, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    sub_tasks.append(f"Sub-task 4_1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {
        "thinking": thinking4_1,
        "answer": answer4_1
    }
    logs.append(subtask_desc4_1)
    print("Step 4_1: ", sub_tasks[-1])
    
    cot_instruction_4_2 = "Sub-task 4_2: Evaluate the feasibility and mechanistic details of each pathway identified in Sub-task 4_1, considering stereochemistry, reaction conditions, and known product types to narrow down the most plausible product formation route."
    cot_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_2 = self.max_round
    cot_inputs_4_2 = [taskInfo, thinking2, answer2, thinking4_1, answer4_1]
    subtask_desc4_2 = {
        "subtask_id": "subtask_4_2",
        "instruction": cot_instruction_4_2,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 4_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_instruction_4_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, evaluating pathway feasibility, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    for i in range(N_max_4_2):
        feedback, correct = await critic_agent_4_2([taskInfo, thinking4_2, answer4_2], "please review the pathway feasibility evaluation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_2.id}, providing feedback on pathway evaluation, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_2.extend([thinking4_2, answer4_2, feedback])
        thinking4_2, answer4_2 = await cot_agent_4_2(cot_inputs_4_2, cot_instruction_4_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_2.id}, refining pathway evaluation, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 4_2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {
        "thinking": thinking4_2,
        "answer": answer4_2
    }
    logs.append(subtask_desc4_2)
    print("Step 4_2: ", sub_tasks[-1])
    
    cot_instruction_4_3 = "Sub-task 4_3: Predict the final product B of the second reaction based on the selected pericyclic pathway from Sub-task 4_2, ensuring correct stereochemical assignments and product structure consistent with thermal diyne rearrangements."
    cot_agent_4_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_3 = self.max_round
    cot_inputs_4_3 = [taskInfo, thinking2, answer2, thinking4_1, answer4_1, thinking4_2, answer4_2]
    subtask_desc4_3 = {
        "subtask_id": "subtask_4_3",
        "instruction": cot_instruction_4_3,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 4_1", "thinking and answer of subtask 4_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking4_3, answer4_3 = await cot_agent_4_3(cot_inputs_4_3, cot_instruction_4_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_3.id}, predicting product B, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    for i in range(N_max_4_3):
        feedback, correct = await critic_agent_4_3([taskInfo, thinking4_3, answer4_3], "please review the product B prediction and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_3.id}, providing feedback on product B prediction, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_3.extend([thinking4_3, answer4_3, feedback])
        thinking4_3, answer4_3 = await cot_agent_4_3(cot_inputs_4_3, cot_instruction_4_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_3.id}, refining product B prediction, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    sub_tasks.append(f"Sub-task 4_3 output: thinking - {thinking4_3.content}; answer - {answer4_3.content}")
    subtask_desc4_3['response'] = {
        "thinking": thinking4_3,
        "answer": answer4_3
    }
    logs.append(subtask_desc4_3)
    print("Step 4_3: ", sub_tasks[-1])
    
    cot_instruction_5_1 = "Sub-task 5_1: For the third reaction (2-((vinyloxy)methyl)but-1-ene + Heat), identify the reactive intermediates and mechanistic steps involved in the Claisen rearrangement or related sigmatropic rearrangements under thermal conditions, based on Sub-task 2 outputs."
    cot_agent_5_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5_1 = {
        "subtask_id": "subtask_5_1",
        "instruction": cot_instruction_5_1,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking5_1, answer5_1 = await cot_agent_5_1([taskInfo, thinking2, answer2], cot_instruction_5_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5_1.id}, identifying reactive intermediates for third reaction, thinking: {thinking5_1.content}; answer: {answer5_1.content}")
    sub_tasks.append(f"Sub-task 5_1 output: thinking - {thinking5_1.content}; answer - {answer5_1.content}")
    subtask_desc5_1['response'] = {
        "thinking": thinking5_1,
        "answer": answer5_1
    }
    logs.append(subtask_desc5_1)
    print("Step 5_1: ", sub_tasks[-1])
    
    cot_instruction_5_2 = "Sub-task 5_2: Predict the final product C of the third reaction by applying the Claisen rearrangement mechanism, ensuring correct structural and stereochemical features of the product, and distinguishing between aldehyde and alcohol functional groups as possible outcomes."
    cot_agent_5_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5_2 = self.max_round
    cot_inputs_5_2 = [taskInfo, thinking2, answer2, thinking5_1, answer5_1]
    subtask_desc5_2 = {
        "subtask_id": "subtask_5_2",
        "instruction": cot_instruction_5_2,
        "context": ["user query", "thinking and answer of subtask 2", "thinking and answer of subtask 5_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking5_2, answer5_2 = await cot_agent_5_2(cot_inputs_5_2, cot_instruction_5_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5_2.id}, predicting product C, thinking: {thinking5_2.content}; answer: {answer5_2.content}")
    for i in range(N_max_5_2):
        feedback, correct = await critic_agent_5_2([taskInfo, thinking5_2, answer5_2], "please review the product C prediction and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5_2.id}, providing feedback on product C prediction, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5_2.extend([thinking5_2, answer5_2, feedback])
        thinking5_2, answer5_2 = await cot_agent_5_2(cot_inputs_5_2, cot_instruction_5_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5_2.id}, refining product C prediction, thinking: {thinking5_2.content}; answer: {answer5_2.content}")
    sub_tasks.append(f"Sub-task 5_2 output: thinking - {thinking5_2.content}; answer - {answer5_2.content}")
    subtask_desc5_2['response'] = {
        "thinking": thinking5_2,
        "answer": answer5_2
    }
    logs.append(subtask_desc5_2)
    print("Step 5_2: ", sub_tasks[-1])
    
    debate_instruction_6 = "Sub-task 6: Compare the predicted products A, B, and C from Sub-tasks 3_3, 4_3, and 5_2 with the given multiple-choice options, performing cross-validation to identify the set of products that best matches mechanistic reasoning and structural correctness."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking and answer of subtask 3_3", "thinking and answer of subtask 4_3", "thinking and answer of subtask 5_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking3_3, answer3_3, thinking4_3, answer4_3, thinking5_2, answer5_2], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking3_3, answer3_3, thinking4_3, answer4_3, thinking5_2, answer5_2] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products and options, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 7: Select and return the correct multiple-choice answer (A, B, C, or D) based on the validated comparison in Sub-task 6, providing justification grounded in detailed mechanistic analysis and product structure consistency.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": "Sub-task 7: Select and return the correct multiple-choice answer (A, B, C, or D) based on the validated comparison in Sub-task 6, providing justification grounded in detailed mechanistic analysis and product structure consistency.",
        "context": ["user query", "thinking and answer of subtask 6"],
        "agent_collaboration": "Final Decision"
    }
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
