async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Characterize the differences between PFA fixation alone and PFA+DSG dual fixation in ChIP-seq experiments, focusing on how these fixation chemistries affect protein-DNA crosslinking efficiency, epitope accessibility, and the potential impact on ChIP peak detection."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, characterizing fixation differences, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_instruction_2a = "Sub-task 2a: Retrieve and compile empirical evidence from published ChIP-seq datasets on the genomic distribution of IKAROS binding sites in human B cells, including quantitative data on peak enrichment at promoters, enhancers, repeats, and introns. Cite relevant studies or datasets."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3)
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2a, answer2a = await cot_agent_2a([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, retrieving empirical IKAROS binding data, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])
    cot_instruction_2b = "Sub-task 2b: Summarize the proportion of IKAROS ChIP-seq peaks located at active promoters/enhancers versus repeats and introns, based on the compiled empirical data from Sub-task 2a, highlighting any uncertainties or variability."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3)
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "CoT"
    }
    thinking2b, answer2b = await cot_agent_2b([taskInfo, thinking1, answer1, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, summarizing IKAROS peak proportions, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])
    cot_sc_instruction_2c = "Sub-task 2c: Perform a Self-Consistency Chain-of-Thought analysis to generate multiple hypotheses about IKAROS binding distribution and select the most supported hypothesis grounded in empirical evidence from Sub-tasks 2a and 2b."
    N = self.max_sc
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2c, answer2c = await cot_agents_2c[i]([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, generating hypotheses on IKAROS binding, thinking: {thinking2c.content}; answer: {answer2c.content}")
        possible_answers_2c.append(answer2c.content)
        thinkingmapping_2c[answer2c.content] = thinking2c
        answermapping_2c[answer2c.content] = answer2c
    answer2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking2c = thinkingmapping_2c[answer2c_content]
    answer2c = answermapping_2c[answer2c_content]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {
        "thinking": thinking2c,
        "answer": answer2c
    }
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])
    debate_instruction_2d = "Sub-task 2d: Conduct a Debate step to critically evaluate and validate the assumptions and conclusions about IKAROS binding site distribution from Sub-task 2c, incorporating alternative perspectives and potential data limitations."
    debate_agents_2d = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2d = self.max_round
    all_thinking2d = [[] for _ in range(N_max_2d)]
    all_answer2d = [[] for _ in range(N_max_2d)]
    subtask_desc2d = {
        "subtask_id": "subtask_2d",
        "instruction": debate_instruction_2d,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2d):
        for i, agent in enumerate(debate_agents_2d):
            if r == 0:
                thinking2d, answer2d = await agent([taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c], debate_instruction_2d, r, is_sub_task=True)
            else:
                input_infos_2d = [taskInfo, thinking1, answer1, thinking2a, answer2a, thinking2b, answer2b, thinking2c, answer2c] + all_thinking2d[r-1] + all_answer2d[r-1]
                thinking2d, answer2d = await agent(input_infos_2d, debate_instruction_2d, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating IKAROS binding assumptions, thinking: {thinking2d.content}; answer: {answer2d.content}")
            all_thinking2d[r].append(thinking2d)
            all_answer2d[r].append(answer2d)
    final_decision_agent_2d = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2d, answer2d = await final_decision_agent_2d([taskInfo] + all_thinking2d[-1] + all_answer2d[-1], "Sub-task 2d: Make final decision on validated IKAROS binding site distribution.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing IKAROS binding distribution, thinking: {thinking2d.content}; answer: {answer2d.content}")
    sub_tasks.append(f"Sub-task 2d output: thinking - {thinking2d.content}; answer - {answer2d.content}")
    subtask_desc2d['response'] = {
        "thinking": thinking2d,
        "answer": answer2d
    }
    logs.append(subtask_desc2d)
    print("Step 2d: ", sub_tasks[-1])
    cot_instruction_3a = "Sub-task 3a: Analyze how the addition of DSG to PFA fixation might alter ChIP-seq peak detection for IKAROS by exploring technical factors such as epitope masking, antibody accessibility, chromatin compaction, and crosslinking efficiency at different genomic regions, based on Sub-tasks 1 and 2d outputs."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2d", "answer of subtask 2d"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking1, answer1, thinking2d, answer2d], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, analyzing DSG fixation impact technical factors, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_instruction_3b = "Sub-task 3b: Integrate prior knowledge and published observations on how DSG crosslinking affects epitope accessibility at promoters, enhancers, and heterochromatin to inform the interpretation of disappearing IKAROS ChIP-seq peaks, based on Sub-task 3a output."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, integrating DSG crosslinking prior knowledge, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    debate_instruction_4 = "Sub-task 4: Synthesize insights from all previous subtasks to determine the most probable genomic locations of disappearing IKAROS ChIP-seq peaks when switching from PFA to PFA+DSG fixation, selecting the best answer among the provided multiple-choice options with biological justification."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2d", "answer of subtask 2d", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking1, answer1, thinking2d, answer2d, thinking3a, answer3a, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking1, answer1, thinking2d, answer2d, thinking3a, answer3a, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, synthesizing disappearing peak locations, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the most probable genomic locations of disappearing IKAROS ChIP-seq peaks.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on disappearing peak locations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_reflect_instruction_5 = "Sub-task 5: Perform a Reflexion checkpoint to critically assess whether the final conclusion about disappearing IKAROS ChIP-seq peaks aligns with established biological knowledge and empirical data, flagging any discrepancies or uncertainties before finalizing the answer."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, assessing final conclusion validity, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the final conclusion on disappearing IKAROS peaks for biological consistency and empirical support.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining final conclusion, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs