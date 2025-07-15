async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks=[]
    agents=[]
    logs=[]

    # Sub-task 1: SC_CoT extract and classify inputs
    cot_sc_instruction = "Sub-task 1: Extract and classify all inputs: identify the starting material, the target molecule, and each reagent’s role (type of reaction and directing effects)."
    N=self.max_sc
    cot_agents=[LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_think1=[]
    possible_ans1=[]
    desc1={"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for ag in cot_agents:
        thinking,answer=await ag([taskInfo],cot_sc_instruction,is_sub_task=True)
        agents.append(f"CoT-SC agent {ag.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think1.append(thinking)
        possible_ans1.append(answer)
    final_decision=LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking1,answer1=await final_decision([taskInfo]+possible_think1+possible_ans1,"Sub-task 1: Synthesize and choose the most consistent classification for inputs.",is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    desc1['response']={"thinking":thinking1,"answer":answer1}
    logs.append(desc1)
    print("Step 1: ",sub_tasks[-1])

    # Sub-task 2: SC_CoT identify ambiguities
    cot_sc_instruction2 = "Sub-task 2: Identify ambiguities or missing details in reagents and transformations to ensure clarity in subsequent planning."
    cot_agents2=[LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_think2=[]
    possible_ans2=[]
    desc2={"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query",thinking1,answer1],"agent_collaboration":"SC_CoT"}
    for ag in cot_agents2:
        thinking,answer=await ag([taskInfo,thinking1,answer1],cot_sc_instruction2,is_sub_task=True)
        agents.append(f"CoT-SC agent {ag.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think2.append(thinking)
        possible_ans2.append(answer)
    final2=LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking2,answer2=await final2([taskInfo,thinking1,answer1]+possible_think2+possible_ans2,"Sub-task 2: Synthesize and choose the most consistent list of ambiguities.",is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    desc2['response']={"thinking":thinking2,"answer":answer2}
    logs.append(desc2)
    print("Step 2: ",sub_tasks[-1])

    # Sub-task 3: Debate outline optimal sequence
    debate_instr="Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction3="Sub-task 3: Outline an optimal sequence of reactions—choosing when to install tert-butyl, nitro, and ethoxy groups, and when to do reductions/diazotizations—for correct regiochemistry and high yield."+debate_instr
    debate_agents=[LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    rounds=self.max_round
    all_think3=[[] for _ in range(rounds)]
    all_ans3=[[] for _ in range(rounds)]
    desc3={"subtask_id":"subtask_3","instruction":debate_instruction3,"context":["user query",thinking2,answer2],"agent_collaboration":"Debate"}
    for r in range(rounds):
        for ag in debate_agents:
            if r==0:
                thinking,answer=await ag([taskInfo,thinking2,answer2],debate_instruction3,r,is_sub_task=True)
            else:
                inputs=[taskInfo,thinking2,answer2]+all_think3[r-1]+all_ans3[r-1]
                thinking,answer=await ag(inputs,debate_instruction3,r,is_sub_task=True)
            agents.append(f"Debate agent {ag.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_think3[r].append(thinking)
            all_ans3[r].append(answer)
    final3=LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking3,answer3=await final3([taskInfo,thinking2,answer2]+all_think3[-1]+all_ans3[-1],"Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    desc3['response']={"thinking":thinking3,"answer":answer3}
    logs.append(desc3)
    print("Step 3: ",sub_tasks[-1])

    # Sub-task 4: SC_CoT compare to choices
    cot_sc_instruction4 = "Sub-task 4: Compare the proposed ideal sequence to each of the four provided choices, evaluating which option matches step-by-step."
    cot_agents4=[LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_think4=[]
    possible_ans4=[]
    desc4={"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query",thinking3,answer3],"agent_collaboration":"SC_CoT"}
    for ag in cot_agents4:
        thinking,answer=await ag([taskInfo,thinking3,answer3],cot_sc_instruction4,is_sub_task=True)
        agents.append(f"CoT-SC agent {ag.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_think4.append(thinking)
        possible_ans4.append(answer)
    final4=LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking4,answer4=await final4([taskInfo,thinking3,answer3]+possible_think4+possible_ans4,"Sub-task 4: Synthesize and choose the most consistent comparison.",is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    desc4['response']={"thinking":thinking4,"answer":answer4}
    logs.append(desc4)
    print("Step 4: ",sub_tasks[-1])

    # Sub-task 5: CoT select correct sequence
    cot_instruction5="Sub-task 5: Select the correct multiple-choice sequence and provide a concise justification based on regiochemical control and reaction order."
    cot_agent5=LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    thinking5,answer5=await cot_agent5([taskInfo,thinking4,answer4],cot_instruction5,is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    desc5={"subtask_id":"subtask_5","instruction":cot_instruction5,"context":["user query",thinking4,answer4],"agent_collaboration":"CoT","response":{"thinking":thinking5,"answer":answer5}
    logs.append(desc5)
    print("Step 5: ",sub_tasks[-1])

    final_answer=await self.make_final_answer(thinking5,answer5,sub_tasks,agents)
    return final_answer,logs