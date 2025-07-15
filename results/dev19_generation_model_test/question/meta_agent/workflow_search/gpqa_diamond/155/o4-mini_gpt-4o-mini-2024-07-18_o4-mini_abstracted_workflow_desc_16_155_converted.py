async def forward_155(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: Determine stereochemical outcome of syn epoxidation
    cot_sc_instruction = "Sub-task 1: Based on the query, determine how syn epoxidation of (E)-oct-4-ene and (Z)-oct-4-ene with mCPBA proceeds in terms of absolute configurations of the epoxides."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents1:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision1 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo]+possible_thinkings1+possible_answers1,
        "Sub-task 1: Synthesize the most consistent stereochemical outcome for both epoxidations.",is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: List and classify distinct stereoisomeric epoxide products
    cot_sc_instruction = "Sub-task 2: Given the stereochemical outcomes from Sub-task 1, list all distinct stereoisomeric epoxide structures and classify which are enantiomeric pairs versus meso."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision2 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1]+possible_thinkings2+possible_answers2,
        "Sub-task 2: Synthesize the distinct stereoisomer count and classification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: Predict peaks on standard HPLC
    cot_sc_instruction = "Sub-task 3: Using the stereoisomer set from Sub-task 2, predict how many peaks will appear on an achiral reverse-phase HPLC assuming perfect resolution (diastereomers separate, enantiomers coelute)."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction,"context":["user query","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decision3 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2]+possible_thinkings3+possible_answers3,
        "Sub-task 3: Synthesize the predicted number of achiral HPLC peaks.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: Predict peaks on chiral HPLC
    cot_sc_instruction = "Sub-task 4: Using the stereoisomer set from Sub-task 2, predict how many peaks will appear on a chiral HPLC assuming perfect resolution (all stereoisomers separate)."
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction,"context":["user query","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents4:
        thinking, answer = await agent([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings4.append(thinking)
        possible_answers4.append(answer)
    final_decision4 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking4, answer4 = await final_decision4([taskInfo, thinking2, answer2]+possible_thinkings4+possible_answers4,
        "Sub-task 4: Synthesize the predicted number of chiral HPLC peaks.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Sub-task 5: Match predicted counts to the multiple-choice answer with Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 5: Based on the predicted peak counts from Sub-tasks 3 and 4, choose which multiple-choice option matches. " + reflect_inst
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    cot_inputs = [taskInfo, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction,"context":["user query","answers of subtask 3","answers of subtask 4"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for _ in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent(cot_inputs, cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refined thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
